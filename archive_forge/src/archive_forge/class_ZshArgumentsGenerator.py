import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
class ZshArgumentsGenerator:
    """
    Generate a call to the zsh _arguments completion function
    based on data in a usage.Options instance

    The first three instance variables are populated based on constructor
    arguments. The remaining non-constructor variables are populated by this
    class with data gathered from the C{Options} instance passed in, and its
    base classes.

    @type options: L{twisted.python.usage.Options}
    @ivar options: The L{twisted.python.usage.Options} instance to generate for

    @type cmdName: C{str}
    @ivar cmdName: The name of the command we're generating completions for.

    @type file: C{file}
    @ivar file: The C{file} to write the completion function to.  The C{file}
        must have L{bytes} I/O semantics.

    @type descriptions: C{dict}
    @ivar descriptions: A dict mapping long option names to alternate
        descriptions. When this variable is defined, the descriptions
        contained here will override those descriptions provided in the
        optFlags and optParameters variables.

    @type multiUse: C{list}
    @ivar multiUse: An iterable containing those long option names which may
        appear on the command line more than once. By default, options will
        only be completed one time.

    @type mutuallyExclusive: C{list} of C{tuple}
    @ivar mutuallyExclusive: A sequence of sequences, with each sub-sequence
        containing those long option names that are mutually exclusive. That is,
        those options that cannot appear on the command line together.

    @type optActions: C{dict}
    @ivar optActions: A dict mapping long option names to shell "actions".
        These actions define what may be completed as the argument to the
        given option, and should be given as instances of
        L{twisted.python.usage.Completer}.

        Callables may instead be given for the values in this dict. The
        callable should accept no arguments, and return a C{Completer}
        instance used as the action.

    @type extraActions: C{list} of C{twisted.python.usage.Completer}
    @ivar extraActions: Extra arguments are those arguments typically
        appearing at the end of the command-line, which are not associated
        with any particular named option. That is, the arguments that are
        given to the parseArgs() method of your usage.Options subclass.
    """

    def __init__(self, options, cmdName, file):
        self.options = options
        self.cmdName = cmdName
        self.file = file
        self.descriptions = {}
        self.multiUse = set()
        self.mutuallyExclusive = []
        self.optActions = {}
        self.extraActions = []
        for cls in reversed(inspect.getmro(options.__class__)):
            data = getattr(cls, 'compData', None)
            if data:
                self.descriptions.update(data.descriptions)
                self.optActions.update(data.optActions)
                self.multiUse.update(data.multiUse)
                self.mutuallyExclusive.extend(data.mutuallyExclusive)
                if data.extraActions:
                    self.extraActions = data.extraActions
        aCL = reflect.accumulateClassList
        optFlags: List[List[object]] = []
        optParams: List[List[object]] = []
        aCL(options.__class__, 'optFlags', optFlags)
        aCL(options.__class__, 'optParameters', optParams)
        for i, optList in enumerate(optFlags):
            if len(optList) != 3:
                optFlags[i] = util.padTo(3, optList)
        for i, optList in enumerate(optParams):
            if len(optList) != 5:
                optParams[i] = util.padTo(5, optList)
        self.optFlags = optFlags
        self.optParams = optParams
        paramNameToDefinition = {}
        for optList in optParams:
            paramNameToDefinition[optList[0]] = optList[1:]
        self.paramNameToDefinition = paramNameToDefinition
        flagNameToDefinition = {}
        for optList in optFlags:
            flagNameToDefinition[optList[0]] = optList[1:]
        self.flagNameToDefinition = flagNameToDefinition
        allOptionsNameToDefinition = {}
        allOptionsNameToDefinition.update(paramNameToDefinition)
        allOptionsNameToDefinition.update(flagNameToDefinition)
        self.allOptionsNameToDefinition = allOptionsNameToDefinition
        self.addAdditionalOptions()
        self.verifyZshNames()
        self.excludes = self.makeExcludesDict()

    def write(self):
        """
        Write the zsh completion code to the file given to __init__
        @return: L{None}
        """
        self.writeHeader()
        self.writeExtras()
        self.writeOptions()
        self.writeFooter()

    def writeHeader(self):
        """
        This is the start of the code that calls _arguments
        @return: L{None}
        """
        self.file.write(b'#compdef ' + self.cmdName.encode('utf-8') + b'\n\n_arguments -s -A "-*" \\\n')

    def writeOptions(self):
        """
        Write out zsh code for each option in this command
        @return: L{None}
        """
        optNames = list(self.allOptionsNameToDefinition.keys())
        optNames.sort()
        for longname in optNames:
            self.writeOpt(longname)

    def writeExtras(self):
        """
        Write out completion information for extra arguments appearing on the
        command-line. These are extra positional arguments not associated
        with a named option. That is, the stuff that gets passed to
        Options.parseArgs().

        @return: L{None}

        @raise ValueError: If C{Completer} with C{repeat=True} is found and
            is not the last item in the C{extraActions} list.
        """
        for i, action in enumerate(self.extraActions):
            if action._repeat and i != len(self.extraActions) - 1:
                raise ValueError('Completer with repeat=True must be last item in Options.extraActions')
            self.file.write(escape(action._shellCode('', usage._ZSH)).encode('utf-8'))
            self.file.write(b' \\\n')

    def writeFooter(self):
        """
        Write the last bit of code that finishes the call to _arguments
        @return: L{None}
        """
        self.file.write(b'&& return 0\n')

    def verifyZshNames(self):
        """
        Ensure that none of the option names given in the metadata are typoed
        @return: L{None}
        @raise ValueError: If unknown option names have been found.
        """

        def err(name):
            raise ValueError('Unknown option name "%s" found while\nexamining Completions instances on %s' % (name, self.options))
        for name in itertools.chain(self.descriptions, self.optActions, self.multiUse):
            if name not in self.allOptionsNameToDefinition:
                err(name)
        for seq in self.mutuallyExclusive:
            for name in seq:
                if name not in self.allOptionsNameToDefinition:
                    err(name)

    def excludeStr(self, longname, buildShort=False):
        """
        Generate an "exclusion string" for the given option

        @type longname: C{str}
        @param longname: The long option name (e.g. "verbose" instead of "v")

        @type buildShort: C{bool}
        @param buildShort: May be True to indicate we're building an excludes
            string for the short option that corresponds to the given long opt.

        @return: The generated C{str}
        """
        if longname in self.excludes:
            exclusions = self.excludes[longname].copy()
        else:
            exclusions = set()
        if longname not in self.multiUse:
            if buildShort is False:
                short = self.getShortOption(longname)
                if short is not None:
                    exclusions.add(short)
            else:
                exclusions.add(longname)
        if not exclusions:
            return ''
        strings = []
        for optName in exclusions:
            if len(optName) == 1:
                strings.append('-' + optName)
            else:
                strings.append('--' + optName)
        strings.sort()
        return '(%s)' % ' '.join(strings)

    def makeExcludesDict(self) -> Dict[str, Set[str]]:
        """
        @return: A C{dict} that maps each option name appearing in
            self.mutuallyExclusive to a set of those option names that is it
            mutually exclusive with (can't appear on the cmd line with).
        """
        longToShort = {}
        for optList in itertools.chain(self.optParams, self.optFlags):
            if optList[1] != None:
                longToShort[optList[0]] = optList[1]
        excludes: Dict[str, Set[str]] = {}
        for lst in self.mutuallyExclusive:
            for i, longname in enumerate(lst):
                tmp = set(lst[:i] + lst[i + 1:])
                for name in tmp.copy():
                    if name in longToShort:
                        tmp.add(longToShort[name])
                if longname in excludes:
                    excludes[longname] = excludes[longname].union(tmp)
                else:
                    excludes[longname] = tmp
        return excludes

    def writeOpt(self, longname):
        """
        Write out the zsh code for the given argument. This is just part of the
        one big call to _arguments

        @type longname: C{str}
        @param longname: The long option name (e.g. "verbose" instead of "v")

        @return: L{None}
        """
        if longname in self.flagNameToDefinition:
            longField = '--%s' % longname
        else:
            longField = '--%s=' % longname
        short = self.getShortOption(longname)
        if short != None:
            shortField = '-' + short
        else:
            shortField = ''
        descr = self.getDescription(longname)
        descriptionField = descr.replace('[', '\\[')
        descriptionField = descriptionField.replace(']', '\\]')
        descriptionField = '[%s]' % descriptionField
        actionField = self.getAction(longname)
        if longname in self.multiUse:
            multiField = '*'
        else:
            multiField = ''
        longExclusionsField = self.excludeStr(longname)
        if short:
            shortExclusionsField = self.excludeStr(longname, buildShort=True)
            self.file.write(escape('%s%s%s%s%s' % (shortExclusionsField, multiField, shortField, descriptionField, actionField)).encode('utf-8'))
            self.file.write(b' \\\n')
        self.file.write(escape('%s%s%s%s%s' % (longExclusionsField, multiField, longField, descriptionField, actionField)).encode('utf-8'))
        self.file.write(b' \\\n')

    def getAction(self, longname):
        """
        Return a zsh "action" string for the given argument
        @return: C{str}
        """
        if longname in self.optActions:
            if callable(self.optActions[longname]):
                action = self.optActions[longname]()
            else:
                action = self.optActions[longname]
            return action._shellCode(longname, usage._ZSH)
        if longname in self.paramNameToDefinition:
            return f':{longname}:_files'
        return ''

    def getDescription(self, longname):
        """
        Return the description to be used for this argument
        @return: C{str}
        """
        if longname in self.descriptions:
            return self.descriptions[longname]
        try:
            descr = self.flagNameToDefinition[longname][1]
        except KeyError:
            try:
                descr = self.paramNameToDefinition[longname][2]
            except KeyError:
                descr = None
        if descr is not None:
            return descr
        longMangled = longname.replace('-', '_')
        obj = getattr(self.options, 'opt_%s' % longMangled, None)
        if obj is not None:
            descr = descrFromDoc(obj)
            if descr is not None:
                return descr
        return longname

    def getShortOption(self, longname):
        """
        Return the short option letter or None
        @return: C{str} or L{None}
        """
        optList = self.allOptionsNameToDefinition[longname]
        return optList[0] or None

    def addAdditionalOptions(self) -> None:
        """
        Add additional options to the optFlags and optParams lists.
        These will be defined by 'opt_foo' methods of the Options subclass
        @return: L{None}
        """
        methodsDict: Dict[str, MethodType] = {}
        reflect.accumulateMethods(self.options, methodsDict, 'opt_')
        methodToShort = {}
        for name in methodsDict.copy():
            if len(name) == 1:
                methodToShort[methodsDict[name]] = name
                del methodsDict[name]
        for methodName, methodObj in methodsDict.items():
            longname = methodName.replace('_', '-')
            if longname in self.allOptionsNameToDefinition:
                continue
            descr = self.getDescription(longname)
            short = None
            if methodObj in methodToShort:
                short = methodToShort[methodObj]
            reqArgs = methodObj.__func__.__code__.co_argcount
            if reqArgs == 2:
                self.optParams.append([longname, short, None, descr])
                self.paramNameToDefinition[longname] = [short, None, descr]
                self.allOptionsNameToDefinition[longname] = [short, None, descr]
            else:
                self.optFlags.append([longname, short, descr])
                self.flagNameToDefinition[longname] = [short, descr]
                self.allOptionsNameToDefinition[longname] = [short, None, descr]