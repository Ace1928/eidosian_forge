from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
class _ModuleContainer(object):
    """
    Encapsulates the capability to contain Module objects.

    @note: Labels are an approximated way of referencing memory locations
        across different executions of the same process, or different processes
        with common modules. They are not meant to be perfectly unique, and
        some errors may occur when multiple modules with the same name are
        loaded, or when module filenames can't be retrieved.

    @group Modules snapshot:
        scan_modules,
        get_module, get_module_bases, get_module_count,
        get_module_at_address, get_module_by_name,
        has_module, iter_modules, iter_module_addresses,
        clear_modules

    @group Labels:
        parse_label, split_label, sanitize_label, resolve_label,
        resolve_label_components, get_label_at_address, split_label_strict,
        split_label_fuzzy

    @group Symbols:
        load_symbols, unload_symbols, get_symbols, iter_symbols,
        resolve_symbol, get_symbol_at_address

    @group Debugging:
        is_system_defined_breakpoint, get_system_breakpoint,
        get_user_breakpoint, get_breakin_breakpoint,
        get_wow64_system_breakpoint, get_wow64_user_breakpoint,
        get_wow64_breakin_breakpoint, get_break_on_error_ptr
    """

    def __init__(self):
        self.__moduleDict = dict()
        self.__system_breakpoints = dict()
        self.split_label = self.__use_fuzzy_mode

    def __initialize_snapshot(self):
        """
        Private method to automatically initialize the snapshot
        when you try to use it without calling any of the scan_*
        methods first. You don't need to call this yourself.
        """
        if not self.__moduleDict:
            try:
                self.scan_modules()
            except WindowsError:
                pass

    def __contains__(self, anObject):
        """
        @type  anObject: L{Module}, int
        @param anObject:
            - C{Module}: Module object to look for.
            - C{int}: Base address of the DLL to look for.

        @rtype:  bool
        @return: C{True} if the snapshot contains
            a L{Module} object with the same base address.
        """
        if isinstance(anObject, Module):
            anObject = anObject.lpBaseOfDll
        return self.has_module(anObject)

    def __iter__(self):
        """
        @see:    L{iter_modules}
        @rtype:  dictionary-valueiterator
        @return: Iterator of L{Module} objects in this snapshot.
        """
        return self.iter_modules()

    def __len__(self):
        """
        @see:    L{get_module_count}
        @rtype:  int
        @return: Count of L{Module} objects in this snapshot.
        """
        return self.get_module_count()

    def has_module(self, lpBaseOfDll):
        """
        @type  lpBaseOfDll: int
        @param lpBaseOfDll: Base address of the DLL to look for.

        @rtype:  bool
        @return: C{True} if the snapshot contains a
            L{Module} object with the given base address.
        """
        self.__initialize_snapshot()
        return lpBaseOfDll in self.__moduleDict

    def get_module(self, lpBaseOfDll):
        """
        @type  lpBaseOfDll: int
        @param lpBaseOfDll: Base address of the DLL to look for.

        @rtype:  L{Module}
        @return: Module object with the given base address.
        """
        self.__initialize_snapshot()
        if lpBaseOfDll not in self.__moduleDict:
            msg = 'Unknown DLL base address %s'
            msg = msg % HexDump.address(lpBaseOfDll)
            raise KeyError(msg)
        return self.__moduleDict[lpBaseOfDll]

    def iter_module_addresses(self):
        """
        @see:    L{iter_modules}
        @rtype:  dictionary-keyiterator
        @return: Iterator of DLL base addresses in this snapshot.
        """
        self.__initialize_snapshot()
        return compat.iterkeys(self.__moduleDict)

    def iter_modules(self):
        """
        @see:    L{iter_module_addresses}
        @rtype:  dictionary-valueiterator
        @return: Iterator of L{Module} objects in this snapshot.
        """
        self.__initialize_snapshot()
        return compat.itervalues(self.__moduleDict)

    def get_module_bases(self):
        """
        @see:    L{iter_module_addresses}
        @rtype:  list( int... )
        @return: List of DLL base addresses in this snapshot.
        """
        self.__initialize_snapshot()
        return compat.keys(self.__moduleDict)

    def get_module_count(self):
        """
        @rtype:  int
        @return: Count of L{Module} objects in this snapshot.
        """
        self.__initialize_snapshot()
        return len(self.__moduleDict)

    def get_module_by_name(self, modName):
        """
        @type  modName: int
        @param modName:
            Name of the module to look for, as returned by L{Module.get_name}.
            If two or more modules with the same name are loaded, only one
            of the matching modules is returned.

            You can also pass a full pathname to the DLL file.
            This works correctly even if two modules with the same name
            are loaded from different paths.

        @rtype:  L{Module}
        @return: C{Module} object that best matches the given name.
            Returns C{None} if no C{Module} can be found.
        """
        modName = modName.lower()
        if PathOperations.path_is_absolute(modName):
            for lib in self.iter_modules():
                if modName == lib.get_filename().lower():
                    return lib
            return None
        modDict = [(lib.get_name(), lib) for lib in self.iter_modules()]
        modDict = dict(modDict)
        if modName in modDict:
            return modDict[modName]
        filepart, extpart = PathOperations.split_extension(modName)
        if filepart and extpart:
            if filepart in modDict:
                return modDict[filepart]
        try:
            baseAddress = HexInput.integer(modName)
        except ValueError:
            return None
        if self.has_module(baseAddress):
            return self.get_module(baseAddress)
        return None

    def get_module_at_address(self, address):
        """
        @type  address: int
        @param address: Memory address to query.

        @rtype:  L{Module}
        @return: C{Module} object that best matches the given address.
            Returns C{None} if no C{Module} can be found.
        """
        bases = self.get_module_bases()
        bases.sort()
        bases.append(long(18446744073709551616))
        if address >= bases[0]:
            i = 0
            max_i = len(bases) - 1
            while i < max_i:
                begin, end = bases[i:i + 2]
                if begin <= address < end:
                    module = self.get_module(begin)
                    here = module.is_address_here(address)
                    if here is False:
                        break
                    else:
                        return module
                i = i + 1
        return None

    def scan_modules(self):
        """
        Populates the snapshot with loaded modules.
        """
        dwProcessId = self.get_pid()
        if dwProcessId in (0, 4, 8):
            return
        found_bases = set()
        with win32.CreateToolhelp32Snapshot(win32.TH32CS_SNAPMODULE, dwProcessId) as hSnapshot:
            me = win32.Module32First(hSnapshot)
            while me is not None:
                lpBaseAddress = me.modBaseAddr
                fileName = me.szExePath
                if not fileName:
                    fileName = me.szModule
                    if not fileName:
                        fileName = None
                else:
                    fileName = PathOperations.native_to_win32_pathname(fileName)
                found_bases.add(lpBaseAddress)
                if lpBaseAddress not in self.__moduleDict:
                    aModule = Module(lpBaseAddress, fileName=fileName, SizeOfImage=me.modBaseSize, process=self)
                    self._add_module(aModule)
                else:
                    aModule = self.get_module(lpBaseAddress)
                    if not aModule.fileName:
                        aModule.fileName = fileName
                    if not aModule.SizeOfImage:
                        aModule.SizeOfImage = me.modBaseSize
                    if not aModule.process:
                        aModule.process = self
                me = win32.Module32Next(hSnapshot)
        for base in compat.keys(self.__moduleDict):
            if base not in found_bases:
                self._del_module(base)

    def clear_modules(self):
        """
        Clears the modules snapshot.
        """
        for aModule in compat.itervalues(self.__moduleDict):
            aModule.clear()
        self.__moduleDict = dict()

    @staticmethod
    def parse_label(module=None, function=None, offset=None):
        """
        Creates a label from a module and a function name, plus an offset.

        @warning: This method only creates the label, it doesn't make sure the
            label actually points to a valid memory location.

        @type  module: None or str
        @param module: (Optional) Module name.

        @type  function: None, str or int
        @param function: (Optional) Function name or ordinal.

        @type  offset: None or int
        @param offset: (Optional) Offset value.

            If C{function} is specified, offset from the function.

            If C{function} is C{None}, offset from the module.

        @rtype:  str
        @return:
            Label representing the given function in the given module.

        @raise ValueError:
            The module or function name contain invalid characters.
        """
        try:
            function = '#0x%x' % function
        except TypeError:
            pass
        if module is not None and ('!' in module or '+' in module):
            raise ValueError('Invalid module name: %s' % module)
        if function is not None and ('!' in function or '+' in function):
            raise ValueError('Invalid function name: %s' % function)
        if module:
            if function:
                if offset:
                    label = '%s!%s+0x%x' % (module, function, offset)
                else:
                    label = '%s!%s' % (module, function)
            elif offset:
                label = '%s!0x%x' % (module, offset)
            else:
                label = '%s!' % module
        elif function:
            if offset:
                label = '!%s+0x%x' % (function, offset)
            else:
                label = '!%s' % function
        elif offset:
            label = '0x%x' % offset
        else:
            label = '0x0'
        return label

    @staticmethod
    def split_label_strict(label):
        """
        Splits a label created with L{parse_label}.

        To parse labels with a less strict syntax, use the L{split_label_fuzzy}
        method instead.

        @warning: This method only parses the label, it doesn't make sure the
            label actually points to a valid memory location.

        @type  label: str
        @param label: Label to split.

        @rtype:  tuple( str or None, str or int or None, int or None )
        @return: Tuple containing the C{module} name,
            the C{function} name or ordinal, and the C{offset} value.

            If the label doesn't specify a module,
            then C{module} is C{None}.

            If the label doesn't specify a function,
            then C{function} is C{None}.

            If the label doesn't specify an offset,
            then C{offset} is C{0}.

        @raise ValueError: The label is malformed.
        """
        module = function = None
        offset = 0
        if not label:
            label = '0x0'
        else:
            label = label.replace(' ', '')
            label = label.replace('\t', '')
            label = label.replace('\r', '')
            label = label.replace('\n', '')
            if not label:
                label = '0x0'
        if '!' in label:
            try:
                module, function = label.split('!')
            except ValueError:
                raise ValueError('Malformed label: %s' % label)
            if function:
                if '+' in module:
                    raise ValueError('Malformed label: %s' % label)
                if '+' in function:
                    try:
                        function, offset = function.split('+')
                    except ValueError:
                        raise ValueError('Malformed label: %s' % label)
                    try:
                        offset = HexInput.integer(offset)
                    except ValueError:
                        raise ValueError('Malformed label: %s' % label)
                else:
                    try:
                        offset = HexInput.integer(function)
                        function = None
                    except ValueError:
                        pass
            elif '+' in module:
                try:
                    module, offset = module.split('+')
                except ValueError:
                    raise ValueError('Malformed label: %s' % label)
                try:
                    offset = HexInput.integer(offset)
                except ValueError:
                    raise ValueError('Malformed label: %s' % label)
            else:
                try:
                    offset = HexInput.integer(module)
                    module = None
                except ValueError:
                    pass
            if not module:
                module = None
            if not function:
                function = None
        else:
            try:
                offset = HexInput.integer(label)
            except ValueError:
                if label.startswith('#'):
                    function = label
                    try:
                        HexInput.integer(function[1:])
                    except ValueError:
                        raise ValueError('Ambiguous label: %s' % label)
                else:
                    raise ValueError('Ambiguous label: %s' % label)
        if function and function.startswith('#'):
            try:
                function = HexInput.integer(function[1:])
            except ValueError:
                pass
        if not offset:
            offset = None
        return (module, function, offset)

    def split_label_fuzzy(self, label):
        """
        Splits a label entered as user input.

        It's more flexible in it's syntax parsing than the L{split_label_strict}
        method, as it allows the exclamation mark (B{C{!}}) to be omitted. The
        ambiguity is resolved by searching the modules in the snapshot to guess
        if a label refers to a module or a function. It also tries to rebuild
        labels when they contain hardcoded addresses.

        @warning: This method only parses the label, it doesn't make sure the
            label actually points to a valid memory location.

        @type  label: str
        @param label: Label to split.

        @rtype:  tuple( str or None, str or int or None, int or None )
        @return: Tuple containing the C{module} name,
            the C{function} name or ordinal, and the C{offset} value.

            If the label doesn't specify a module,
            then C{module} is C{None}.

            If the label doesn't specify a function,
            then C{function} is C{None}.

            If the label doesn't specify an offset,
            then C{offset} is C{0}.

        @raise ValueError: The label is malformed.
        """
        module = function = None
        offset = 0
        if not label:
            label = compat.b('0x0')
        else:
            label = label.replace(compat.b(' '), compat.b(''))
            label = label.replace(compat.b('\t'), compat.b(''))
            label = label.replace(compat.b('\r'), compat.b(''))
            label = label.replace(compat.b('\n'), compat.b(''))
            if not label:
                label = compat.b('0x0')
        if compat.b('!') in label:
            return self.split_label_strict(label)
        if compat.b('+') in label:
            try:
                prefix, offset = label.split(compat.b('+'))
            except ValueError:
                raise ValueError('Malformed label: %s' % label)
            try:
                offset = HexInput.integer(offset)
            except ValueError:
                raise ValueError('Malformed label: %s' % label)
            label = prefix
        modobj = self.get_module_by_name(label)
        if modobj:
            module = modobj.get_name()
        else:
            try:
                address = HexInput.integer(label)
                if offset:
                    offset = address + offset
                else:
                    offset = address
                try:
                    new_label = self.get_label_at_address(offset)
                    module, function, offset = self.split_label_strict(new_label)
                except ValueError:
                    pass
            except ValueError:
                function = label
        if function and function.startswith(compat.b('#')):
            try:
                function = HexInput.integer(function[1:])
            except ValueError:
                pass
        if not offset:
            offset = None
        return (module, function, offset)

    @classmethod
    def split_label(cls, label):
        """
Splits a label into it's C{module}, C{function} and C{offset}
components, as used in L{parse_label}.

When called as a static method, the strict syntax mode is used::

    winappdbg.Process.split_label( "kernel32!CreateFileA" )

When called as an instance method, the fuzzy syntax mode is used::

    aProcessInstance.split_label( "CreateFileA" )

@see: L{split_label_strict}, L{split_label_fuzzy}

@type  label: str
@param label: Label to split.

@rtype:  tuple( str or None, str or int or None, int or None )
@return:
    Tuple containing the C{module} name,
    the C{function} name or ordinal, and the C{offset} value.

    If the label doesn't specify a module,
    then C{module} is C{None}.

    If the label doesn't specify a function,
    then C{function} is C{None}.

    If the label doesn't specify an offset,
    then C{offset} is C{0}.

@raise ValueError: The label is malformed.
        """
        return cls.split_label_strict(label)

    def __use_fuzzy_mode(self, label):
        """@see: L{split_label}"""
        return self.split_label_fuzzy(label)

    def sanitize_label(self, label):
        """
        Converts a label taken from user input into a well-formed label.

        @type  label: str
        @param label: Label taken from user input.

        @rtype:  str
        @return: Sanitized label.
        """
        module, function, offset = self.split_label_fuzzy(label)
        label = self.parse_label(module, function, offset)
        return label

    def resolve_label(self, label):
        """
        Resolve the memory address of the given label.

        @note:
            If multiple modules with the same name are loaded,
            the label may be resolved at any of them. For a more precise
            way to resolve functions use the base address to get the L{Module}
            object (see L{Process.get_module}) and then call L{Module.resolve}.

            If no module name is specified in the label, the function may be
            resolved in any loaded module. If you want to resolve all functions
            with that name in all processes, call L{Process.iter_modules} to
            iterate through all loaded modules, and then try to resolve the
            function in each one of them using L{Module.resolve}.

        @type  label: str
        @param label: Label to resolve.

        @rtype:  int
        @return: Memory address pointed to by the label.

        @raise ValueError: The label is malformed or impossible to resolve.
        @raise RuntimeError: Cannot resolve the module or function.
        """
        module, function, offset = self.split_label_fuzzy(label)
        address = self.resolve_label_components(module, function, offset)
        return address

    def resolve_label_components(self, module=None, function=None, offset=None):
        """
        Resolve the memory address of the given module, function and/or offset.

        @note:
            If multiple modules with the same name are loaded,
            the label may be resolved at any of them. For a more precise
            way to resolve functions use the base address to get the L{Module}
            object (see L{Process.get_module}) and then call L{Module.resolve}.

            If no module name is specified in the label, the function may be
            resolved in any loaded module. If you want to resolve all functions
            with that name in all processes, call L{Process.iter_modules} to
            iterate through all loaded modules, and then try to resolve the
            function in each one of them using L{Module.resolve}.

        @type  module: None or str
        @param module: (Optional) Module name.

        @type  function: None, str or int
        @param function: (Optional) Function name or ordinal.

        @type  offset: None or int
        @param offset: (Optional) Offset value.

            If C{function} is specified, offset from the function.

            If C{function} is C{None}, offset from the module.

        @rtype:  int
        @return: Memory address pointed to by the label.

        @raise ValueError: The label is malformed or impossible to resolve.
        @raise RuntimeError: Cannot resolve the module or function.
        """
        address = 0
        if module:
            modobj = self.get_module_by_name(module)
            if not modobj:
                if module == 'main':
                    modobj = self.get_main_module()
                else:
                    raise RuntimeError('Module %r not found' % module)
            if function:
                address = modobj.resolve(function)
                if address is None:
                    address = modobj.resolve_symbol(function)
                    if address is None:
                        if function == 'start':
                            address = modobj.get_entry_point()
                        if address is None:
                            msg = 'Symbol %r not found in module %s'
                            raise RuntimeError(msg % (function, module))
            else:
                address = modobj.get_base()
        elif function:
            for modobj in self.iter_modules():
                address = modobj.resolve(function)
                if address is not None:
                    break
            if address is None:
                if function == 'start':
                    modobj = self.get_main_module()
                    address = modobj.get_entry_point()
                elif function == 'main':
                    modobj = self.get_main_module()
                    address = modobj.get_base()
                else:
                    msg = 'Function %r not found in any module' % function
                    raise RuntimeError(msg)
        if offset:
            address = address + offset
        return address

    def get_label_at_address(self, address, offset=None):
        """
        Creates a label from the given memory address.

        @warning: This method uses the name of the nearest currently loaded
            module. If that module is unloaded later, the label becomes
            impossible to resolve.

        @type  address: int
        @param address: Memory address.

        @type  offset: None or int
        @param offset: (Optional) Offset value.

        @rtype:  str
        @return: Label pointing to the given address.
        """
        if offset:
            address = address + offset
        modobj = self.get_module_at_address(address)
        if modobj:
            label = modobj.get_label_at_address(address)
        else:
            label = self.parse_label(None, None, address)
        return label

    def __get_system_breakpoint(self, label):
        try:
            return self.__system_breakpoints[label]
        except KeyError:
            try:
                address = self.resolve_label(label)
            except Exception:
                return None
            self.__system_breakpoints[label] = address
            return address

    def get_break_on_error_ptr(self):
        """
        @rtype: int
        @return:
            If present, returns the address of the C{g_dwLastErrorToBreakOn}
            global variable for this process. If not, returns C{None}.
        """
        address = self.__get_system_breakpoint('ntdll!g_dwLastErrorToBreakOn')
        if not address:
            address = self.__get_system_breakpoint('kernel32!g_dwLastErrorToBreakOn')
            self.__system_breakpoints['ntdll!g_dwLastErrorToBreakOn'] = address
        return address

    def is_system_defined_breakpoint(self, address):
        """
        @type  address: int
        @param address: Memory address.

        @rtype:  bool
        @return: C{True} if the given address points to a system defined
            breakpoint. System defined breakpoints are hardcoded into
            system libraries.
        """
        if address:
            module = self.get_module_at_address(address)
            if module:
                return module.match_name('ntdll') or module.match_name('kernel32')
        return False

    def get_system_breakpoint(self):
        """
        @rtype:  int or None
        @return: Memory address of the system breakpoint
            within the process address space.
            Returns C{None} on error.
        """
        return self.__get_system_breakpoint('ntdll!DbgBreakPoint')

    def get_user_breakpoint(self):
        """
        @rtype:  int or None
        @return: Memory address of the user breakpoint
            within the process address space.
            Returns C{None} on error.
        """
        return self.__get_system_breakpoint('ntdll!DbgUserBreakPoint')

    def get_breakin_breakpoint(self):
        """
        @rtype:  int or None
        @return: Memory address of the remote breakin breakpoint
            within the process address space.
            Returns C{None} on error.
        """
        return self.__get_system_breakpoint('ntdll!DbgUiRemoteBreakin')

    def get_wow64_system_breakpoint(self):
        """
        @rtype:  int or None
        @return: Memory address of the Wow64 system breakpoint
            within the process address space.
            Returns C{None} on error.
        """
        return self.__get_system_breakpoint('ntdll32!DbgBreakPoint')

    def get_wow64_user_breakpoint(self):
        """
        @rtype:  int or None
        @return: Memory address of the Wow64 user breakpoint
            within the process address space.
            Returns C{None} on error.
        """
        return self.__get_system_breakpoint('ntdll32!DbgUserBreakPoint')

    def get_wow64_breakin_breakpoint(self):
        """
        @rtype:  int or None
        @return: Memory address of the Wow64 remote breakin breakpoint
            within the process address space.
            Returns C{None} on error.
        """
        return self.__get_system_breakpoint('ntdll32!DbgUiRemoteBreakin')

    def load_symbols(self):
        """
        Loads the debugging symbols for all modules in this snapshot.
        Automatically called by L{get_symbols}.
        """
        for aModule in self.iter_modules():
            aModule.load_symbols()

    def unload_symbols(self):
        """
        Unloads the debugging symbols for all modules in this snapshot.
        """
        for aModule in self.iter_modules():
            aModule.unload_symbols()

    def get_symbols(self):
        """
        Returns the debugging symbols for all modules in this snapshot.
        The symbols are automatically loaded when needed.

        @rtype:  list of tuple( str, int, int )
        @return: List of symbols.
            Each symbol is represented by a tuple that contains:
                - Symbol name
                - Symbol memory address
                - Symbol size in bytes
        """
        symbols = list()
        for aModule in self.iter_modules():
            for symbol in aModule.iter_symbols():
                symbols.append(symbol)
        return symbols

    def iter_symbols(self):
        """
        Returns an iterator for the debugging symbols in all modules in this
        snapshot, in no particular order.
        The symbols are automatically loaded when needed.

        @rtype:  iterator of tuple( str, int, int )
        @return: Iterator of symbols.
            Each symbol is represented by a tuple that contains:
                - Symbol name
                - Symbol memory address
                - Symbol size in bytes
        """
        for aModule in self.iter_modules():
            for symbol in aModule.iter_symbols():
                yield symbol

    def resolve_symbol(self, symbol, bCaseSensitive=False):
        """
        Resolves a debugging symbol's address.

        @type  symbol: str
        @param symbol: Name of the symbol to resolve.

        @type  bCaseSensitive: bool
        @param bCaseSensitive: C{True} for case sensitive matches,
            C{False} for case insensitive.

        @rtype:  int or None
        @return: Memory address of symbol. C{None} if not found.
        """
        if bCaseSensitive:
            for SymbolName, SymbolAddress, SymbolSize in self.iter_symbols():
                if symbol == SymbolName:
                    return SymbolAddress
        else:
            symbol = symbol.lower()
            for SymbolName, SymbolAddress, SymbolSize in self.iter_symbols():
                if symbol == SymbolName.lower():
                    return SymbolAddress

    def get_symbol_at_address(self, address):
        """
        Tries to find the closest matching symbol for the given address.

        @type  address: int
        @param address: Memory address to query.

        @rtype: None or tuple( str, int, int )
        @return: Returns a tuple consisting of:
             - Name
             - Address
             - Size (in bytes)
            Returns C{None} if no symbol could be matched.
        """
        found = None
        for SymbolName, SymbolAddress, SymbolSize in self.iter_symbols():
            if SymbolAddress > address:
                continue
            if SymbolAddress == address:
                found = (SymbolName, SymbolAddress, SymbolSize)
                break
            if SymbolAddress < address:
                if found and address - found[1] < address - SymbolAddress:
                    continue
                else:
                    found = (SymbolName, SymbolAddress, SymbolSize)
        return found

    def _add_module(self, aModule):
        """
        Private method to add a module object to the snapshot.

        @type  aModule: L{Module}
        @param aModule: Module object.
        """
        lpBaseOfDll = aModule.get_base()
        aModule.set_process(self)
        self.__moduleDict[lpBaseOfDll] = aModule

    def _del_module(self, lpBaseOfDll):
        """
        Private method to remove a module object from the snapshot.

        @type  lpBaseOfDll: int
        @param lpBaseOfDll: Module base address.
        """
        try:
            aModule = self.__moduleDict[lpBaseOfDll]
            del self.__moduleDict[lpBaseOfDll]
        except KeyError:
            aModule = None
            msg = 'Unknown base address %d' % HexDump.address(lpBaseOfDll)
            warnings.warn(msg, RuntimeWarning)
        if aModule:
            aModule.clear()

    def __add_loaded_module(self, event):
        """
        Private method to automatically add new module objects from debug events.

        @type  event: L{Event}
        @param event: Event object.
        """
        lpBaseOfDll = event.get_module_base()
        hFile = event.get_file_handle()
        if lpBaseOfDll not in self.__moduleDict:
            fileName = event.get_filename()
            if not fileName:
                fileName = None
            if hasattr(event, 'get_start_address'):
                EntryPoint = event.get_start_address()
            else:
                EntryPoint = None
            aModule = Module(lpBaseOfDll, hFile, fileName=fileName, EntryPoint=EntryPoint, process=self)
            self._add_module(aModule)
        else:
            aModule = self.get_module(lpBaseOfDll)
            if not aModule.hFile and hFile not in (None, 0, win32.INVALID_HANDLE_VALUE):
                aModule.hFile = hFile
            if not aModule.process:
                aModule.process = self
            if aModule.EntryPoint is None and hasattr(event, 'get_start_address'):
                aModule.EntryPoint = event.get_start_address()
            if not aModule.fileName:
                fileName = event.get_filename()
                if fileName:
                    aModule.fileName = fileName

    def _notify_create_process(self, event):
        """
        Notify the load of the main module.

        This is done automatically by the L{Debug} class, you shouldn't need
        to call it yourself.

        @type  event: L{CreateProcessEvent}
        @param event: Create process event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        self.__add_loaded_module(event)
        return True

    def _notify_load_dll(self, event):
        """
        Notify the load of a new module.

        This is done automatically by the L{Debug} class, you shouldn't need
        to call it yourself.

        @type  event: L{LoadDLLEvent}
        @param event: Load DLL event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        self.__add_loaded_module(event)
        return True

    def _notify_unload_dll(self, event):
        """
        Notify the release of a loaded module.

        This is done automatically by the L{Debug} class, you shouldn't need
        to call it yourself.

        @type  event: L{UnloadDLLEvent}
        @param event: Unload DLL event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        lpBaseOfDll = event.get_module_base()
        if lpBaseOfDll in self.__moduleDict:
            self._del_module(lpBaseOfDll)
        return True