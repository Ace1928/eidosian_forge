import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, words, \
from pygments.util import get_bool_opt, get_list_opt
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.scanner import Scanner
from pygments.lexers.modula2 import Modula2Lexer
class DelphiLexer(Lexer):
    """
    For `Delphi <http://www.borland.com/delphi/>`_ (Borland Object Pascal),
    Turbo Pascal and Free Pascal source code.

    Additional options accepted:

    `turbopascal`
        Highlight Turbo Pascal specific keywords (default: ``True``).
    `delphi`
        Highlight Borland Delphi specific keywords (default: ``True``).
    `freepascal`
        Highlight Free Pascal specific keywords (default: ``True``).
    `units`
        A list of units that should be considered builtin, supported are
        ``System``, ``SysUtils``, ``Classes`` and ``Math``.
        Default is to consider all of them builtin.
    """
    name = 'Delphi'
    aliases = ['delphi', 'pas', 'pascal', 'objectpascal']
    filenames = ['*.pas', '*.dpr']
    mimetypes = ['text/x-pascal']
    TURBO_PASCAL_KEYWORDS = ('absolute', 'and', 'array', 'asm', 'begin', 'break', 'case', 'const', 'constructor', 'continue', 'destructor', 'div', 'do', 'downto', 'else', 'end', 'file', 'for', 'function', 'goto', 'if', 'implementation', 'in', 'inherited', 'inline', 'interface', 'label', 'mod', 'nil', 'not', 'object', 'of', 'on', 'operator', 'or', 'packed', 'procedure', 'program', 'record', 'reintroduce', 'repeat', 'self', 'set', 'shl', 'shr', 'string', 'then', 'to', 'type', 'unit', 'until', 'uses', 'var', 'while', 'with', 'xor')
    DELPHI_KEYWORDS = ('as', 'class', 'except', 'exports', 'finalization', 'finally', 'initialization', 'is', 'library', 'on', 'property', 'raise', 'threadvar', 'try')
    FREE_PASCAL_KEYWORDS = ('dispose', 'exit', 'false', 'new', 'true')
    BLOCK_KEYWORDS = set(('begin', 'class', 'const', 'constructor', 'destructor', 'end', 'finalization', 'function', 'implementation', 'initialization', 'label', 'library', 'operator', 'procedure', 'program', 'property', 'record', 'threadvar', 'type', 'unit', 'uses', 'var'))
    FUNCTION_MODIFIERS = set(('alias', 'cdecl', 'export', 'inline', 'interrupt', 'nostackframe', 'pascal', 'register', 'safecall', 'softfloat', 'stdcall', 'varargs', 'name', 'dynamic', 'near', 'virtual', 'external', 'override', 'assembler'))
    DIRECTIVES = set(('absolute', 'abstract', 'assembler', 'cppdecl', 'default', 'far', 'far16', 'forward', 'index', 'oldfpccall', 'private', 'protected', 'published', 'public'))
    BUILTIN_TYPES = set(('ansichar', 'ansistring', 'bool', 'boolean', 'byte', 'bytebool', 'cardinal', 'char', 'comp', 'currency', 'double', 'dword', 'extended', 'int64', 'integer', 'iunknown', 'longbool', 'longint', 'longword', 'pansichar', 'pansistring', 'pbool', 'pboolean', 'pbyte', 'pbytearray', 'pcardinal', 'pchar', 'pcomp', 'pcurrency', 'pdate', 'pdatetime', 'pdouble', 'pdword', 'pextended', 'phandle', 'pint64', 'pinteger', 'plongint', 'plongword', 'pointer', 'ppointer', 'pshortint', 'pshortstring', 'psingle', 'psmallint', 'pstring', 'pvariant', 'pwidechar', 'pwidestring', 'pword', 'pwordarray', 'pwordbool', 'real', 'real48', 'shortint', 'shortstring', 'single', 'smallint', 'string', 'tclass', 'tdate', 'tdatetime', 'textfile', 'thandle', 'tobject', 'ttime', 'variant', 'widechar', 'widestring', 'word', 'wordbool'))
    BUILTIN_UNITS = {'System': ('abs', 'acquireexceptionobject', 'addr', 'ansitoutf8', 'append', 'arctan', 'assert', 'assigned', 'assignfile', 'beginthread', 'blockread', 'blockwrite', 'break', 'chdir', 'chr', 'close', 'closefile', 'comptocurrency', 'comptodouble', 'concat', 'continue', 'copy', 'cos', 'dec', 'delete', 'dispose', 'doubletocomp', 'endthread', 'enummodules', 'enumresourcemodules', 'eof', 'eoln', 'erase', 'exceptaddr', 'exceptobject', 'exclude', 'exit', 'exp', 'filepos', 'filesize', 'fillchar', 'finalize', 'findclasshinstance', 'findhinstance', 'findresourcehinstance', 'flush', 'frac', 'freemem', 'get8087cw', 'getdir', 'getlasterror', 'getmem', 'getmemorymanager', 'getmodulefilename', 'getvariantmanager', 'halt', 'hi', 'high', 'inc', 'include', 'initialize', 'insert', 'int', 'ioresult', 'ismemorymanagerset', 'isvariantmanagerset', 'length', 'ln', 'lo', 'low', 'mkdir', 'move', 'new', 'odd', 'olestrtostring', 'olestrtostrvar', 'ord', 'paramcount', 'paramstr', 'pi', 'pos', 'pred', 'ptr', 'pucs4chars', 'random', 'randomize', 'read', 'readln', 'reallocmem', 'releaseexceptionobject', 'rename', 'reset', 'rewrite', 'rmdir', 'round', 'runerror', 'seek', 'seekeof', 'seekeoln', 'set8087cw', 'setlength', 'setlinebreakstyle', 'setmemorymanager', 'setstring', 'settextbuf', 'setvariantmanager', 'sin', 'sizeof', 'slice', 'sqr', 'sqrt', 'str', 'stringofchar', 'stringtoolestr', 'stringtowidechar', 'succ', 'swap', 'trunc', 'truncate', 'typeinfo', 'ucs4stringtowidestring', 'unicodetoutf8', 'uniquestring', 'upcase', 'utf8decode', 'utf8encode', 'utf8toansi', 'utf8tounicode', 'val', 'vararrayredim', 'varclear', 'widecharlentostring', 'widecharlentostrvar', 'widechartostring', 'widechartostrvar', 'widestringtoucs4string', 'write', 'writeln'), 'SysUtils': ('abort', 'addexitproc', 'addterminateproc', 'adjustlinebreaks', 'allocmem', 'ansicomparefilename', 'ansicomparestr', 'ansicomparetext', 'ansidequotedstr', 'ansiextractquotedstr', 'ansilastchar', 'ansilowercase', 'ansilowercasefilename', 'ansipos', 'ansiquotedstr', 'ansisamestr', 'ansisametext', 'ansistrcomp', 'ansistricomp', 'ansistrlastchar', 'ansistrlcomp', 'ansistrlicomp', 'ansistrlower', 'ansistrpos', 'ansistrrscan', 'ansistrscan', 'ansistrupper', 'ansiuppercase', 'ansiuppercasefilename', 'appendstr', 'assignstr', 'beep', 'booltostr', 'bytetocharindex', 'bytetocharlen', 'bytetype', 'callterminateprocs', 'changefileext', 'charlength', 'chartobyteindex', 'chartobytelen', 'comparemem', 'comparestr', 'comparetext', 'createdir', 'createguid', 'currentyear', 'currtostr', 'currtostrf', 'date', 'datetimetofiledate', 'datetimetostr', 'datetimetostring', 'datetimetosystemtime', 'datetimetotimestamp', 'datetostr', 'dayofweek', 'decodedate', 'decodedatefully', 'decodetime', 'deletefile', 'directoryexists', 'diskfree', 'disksize', 'disposestr', 'encodedate', 'encodetime', 'exceptionerrormessage', 'excludetrailingbackslash', 'excludetrailingpathdelimiter', 'expandfilename', 'expandfilenamecase', 'expanduncfilename', 'extractfiledir', 'extractfiledrive', 'extractfileext', 'extractfilename', 'extractfilepath', 'extractrelativepath', 'extractshortpathname', 'fileage', 'fileclose', 'filecreate', 'filedatetodatetime', 'fileexists', 'filegetattr', 'filegetdate', 'fileisreadonly', 'fileopen', 'fileread', 'filesearch', 'fileseek', 'filesetattr', 'filesetdate', 'filesetreadonly', 'filewrite', 'finalizepackage', 'findclose', 'findcmdlineswitch', 'findfirst', 'findnext', 'floattocurr', 'floattodatetime', 'floattodecimal', 'floattostr', 'floattostrf', 'floattotext', 'floattotextfmt', 'fmtloadstr', 'fmtstr', 'forcedirectories', 'format', 'formatbuf', 'formatcurr', 'formatdatetime', 'formatfloat', 'freeandnil', 'getcurrentdir', 'getenvironmentvariable', 'getfileversion', 'getformatsettings', 'getlocaleformatsettings', 'getmodulename', 'getpackagedescription', 'getpackageinfo', 'gettime', 'guidtostring', 'incamonth', 'includetrailingbackslash', 'includetrailingpathdelimiter', 'incmonth', 'initializepackage', 'interlockeddecrement', 'interlockedexchange', 'interlockedexchangeadd', 'interlockedincrement', 'inttohex', 'inttostr', 'isdelimiter', 'isequalguid', 'isleapyear', 'ispathdelimiter', 'isvalidident', 'languages', 'lastdelimiter', 'loadpackage', 'loadstr', 'lowercase', 'msecstotimestamp', 'newstr', 'nextcharindex', 'now', 'outofmemoryerror', 'quotedstr', 'raiselastoserror', 'raiselastwin32error', 'removedir', 'renamefile', 'replacedate', 'replacetime', 'safeloadlibrary', 'samefilename', 'sametext', 'setcurrentdir', 'showexception', 'sleep', 'stralloc', 'strbufsize', 'strbytetype', 'strcat', 'strcharlength', 'strcomp', 'strcopy', 'strdispose', 'strecopy', 'strend', 'strfmt', 'stricomp', 'stringreplace', 'stringtoguid', 'strlcat', 'strlcomp', 'strlcopy', 'strlen', 'strlfmt', 'strlicomp', 'strlower', 'strmove', 'strnew', 'strnextchar', 'strpas', 'strpcopy', 'strplcopy', 'strpos', 'strrscan', 'strscan', 'strtobool', 'strtobooldef', 'strtocurr', 'strtocurrdef', 'strtodate', 'strtodatedef', 'strtodatetime', 'strtodatetimedef', 'strtofloat', 'strtofloatdef', 'strtoint', 'strtoint64', 'strtoint64def', 'strtointdef', 'strtotime', 'strtotimedef', 'strupper', 'supports', 'syserrormessage', 'systemtimetodatetime', 'texttofloat', 'time', 'timestamptodatetime', 'timestamptomsecs', 'timetostr', 'trim', 'trimleft', 'trimright', 'tryencodedate', 'tryencodetime', 'tryfloattocurr', 'tryfloattodatetime', 'trystrtobool', 'trystrtocurr', 'trystrtodate', 'trystrtodatetime', 'trystrtofloat', 'trystrtoint', 'trystrtoint64', 'trystrtotime', 'unloadpackage', 'uppercase', 'widecomparestr', 'widecomparetext', 'widefmtstr', 'wideformat', 'wideformatbuf', 'widelowercase', 'widesamestr', 'widesametext', 'wideuppercase', 'win32check', 'wraptext'), 'Classes': ('activateclassgroup', 'allocatehwnd', 'bintohex', 'checksynchronize', 'collectionsequal', 'countgenerations', 'deallocatehwnd', 'equalrect', 'extractstrings', 'findclass', 'findglobalcomponent', 'getclass', 'groupdescendantswith', 'hextobin', 'identtoint', 'initinheritedcomponent', 'inttoident', 'invalidpoint', 'isuniqueglobalcomponentname', 'linestart', 'objectbinarytotext', 'objectresourcetotext', 'objecttexttobinary', 'objecttexttoresource', 'pointsequal', 'readcomponentres', 'readcomponentresex', 'readcomponentresfile', 'rect', 'registerclass', 'registerclassalias', 'registerclasses', 'registercomponents', 'registerintegerconsts', 'registernoicon', 'registernonactivex', 'smallpoint', 'startclassgroup', 'teststreamformat', 'unregisterclass', 'unregisterclasses', 'unregisterintegerconsts', 'unregistermoduleclasses', 'writecomponentresfile'), 'Math': ('arccos', 'arccosh', 'arccot', 'arccoth', 'arccsc', 'arccsch', 'arcsec', 'arcsech', 'arcsin', 'arcsinh', 'arctan2', 'arctanh', 'ceil', 'comparevalue', 'cosecant', 'cosh', 'cot', 'cotan', 'coth', 'csc', 'csch', 'cycletodeg', 'cycletograd', 'cycletorad', 'degtocycle', 'degtograd', 'degtorad', 'divmod', 'doubledecliningbalance', 'ensurerange', 'floor', 'frexp', 'futurevalue', 'getexceptionmask', 'getprecisionmode', 'getroundmode', 'gradtocycle', 'gradtodeg', 'gradtorad', 'hypot', 'inrange', 'interestpayment', 'interestrate', 'internalrateofreturn', 'intpower', 'isinfinite', 'isnan', 'iszero', 'ldexp', 'lnxp1', 'log10', 'log2', 'logn', 'max', 'maxintvalue', 'maxvalue', 'mean', 'meanandstddev', 'min', 'minintvalue', 'minvalue', 'momentskewkurtosis', 'netpresentvalue', 'norm', 'numberofperiods', 'payment', 'periodpayment', 'poly', 'popnstddev', 'popnvariance', 'power', 'presentvalue', 'radtocycle', 'radtodeg', 'radtograd', 'randg', 'randomrange', 'roundto', 'samevalue', 'sec', 'secant', 'sech', 'setexceptionmask', 'setprecisionmode', 'setroundmode', 'sign', 'simpleroundto', 'sincos', 'sinh', 'slndepreciation', 'stddev', 'sum', 'sumint', 'sumofsquares', 'sumsandsquares', 'syddepreciation', 'tan', 'tanh', 'totalvariance', 'variance')}
    ASM_REGISTERS = set(('ah', 'al', 'ax', 'bh', 'bl', 'bp', 'bx', 'ch', 'cl', 'cr0', 'cr1', 'cr2', 'cr3', 'cr4', 'cs', 'cx', 'dh', 'di', 'dl', 'dr0', 'dr1', 'dr2', 'dr3', 'dr4', 'dr5', 'dr6', 'dr7', 'ds', 'dx', 'eax', 'ebp', 'ebx', 'ecx', 'edi', 'edx', 'es', 'esi', 'esp', 'fs', 'gs', 'mm0', 'mm1', 'mm2', 'mm3', 'mm4', 'mm5', 'mm6', 'mm7', 'si', 'sp', 'ss', 'st0', 'st1', 'st2', 'st3', 'st4', 'st5', 'st6', 'st7', 'xmm0', 'xmm1', 'xmm2', 'xmm3', 'xmm4', 'xmm5', 'xmm6', 'xmm7'))
    ASM_INSTRUCTIONS = set(('aaa', 'aad', 'aam', 'aas', 'adc', 'add', 'and', 'arpl', 'bound', 'bsf', 'bsr', 'bswap', 'bt', 'btc', 'btr', 'bts', 'call', 'cbw', 'cdq', 'clc', 'cld', 'cli', 'clts', 'cmc', 'cmova', 'cmovae', 'cmovb', 'cmovbe', 'cmovc', 'cmovcxz', 'cmove', 'cmovg', 'cmovge', 'cmovl', 'cmovle', 'cmovna', 'cmovnae', 'cmovnb', 'cmovnbe', 'cmovnc', 'cmovne', 'cmovng', 'cmovnge', 'cmovnl', 'cmovnle', 'cmovno', 'cmovnp', 'cmovns', 'cmovnz', 'cmovo', 'cmovp', 'cmovpe', 'cmovpo', 'cmovs', 'cmovz', 'cmp', 'cmpsb', 'cmpsd', 'cmpsw', 'cmpxchg', 'cmpxchg486', 'cmpxchg8b', 'cpuid', 'cwd', 'cwde', 'daa', 'das', 'dec', 'div', 'emms', 'enter', 'hlt', 'ibts', 'icebp', 'idiv', 'imul', 'in', 'inc', 'insb', 'insd', 'insw', 'int', 'int01', 'int03', 'int1', 'int3', 'into', 'invd', 'invlpg', 'iret', 'iretd', 'iretw', 'ja', 'jae', 'jb', 'jbe', 'jc', 'jcxz', 'jcxz', 'je', 'jecxz', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jna', 'jnae', 'jnb', 'jnbe', 'jnc', 'jne', 'jng', 'jnge', 'jnl', 'jnle', 'jno', 'jnp', 'jns', 'jnz', 'jo', 'jp', 'jpe', 'jpo', 'js', 'jz', 'lahf', 'lar', 'lcall', 'lds', 'lea', 'leave', 'les', 'lfs', 'lgdt', 'lgs', 'lidt', 'ljmp', 'lldt', 'lmsw', 'loadall', 'loadall286', 'lock', 'lodsb', 'lodsd', 'lodsw', 'loop', 'loope', 'loopne', 'loopnz', 'loopz', 'lsl', 'lss', 'ltr', 'mov', 'movd', 'movq', 'movsb', 'movsd', 'movsw', 'movsx', 'movzx', 'mul', 'neg', 'nop', 'not', 'or', 'out', 'outsb', 'outsd', 'outsw', 'pop', 'popa', 'popad', 'popaw', 'popf', 'popfd', 'popfw', 'push', 'pusha', 'pushad', 'pushaw', 'pushf', 'pushfd', 'pushfw', 'rcl', 'rcr', 'rdmsr', 'rdpmc', 'rdshr', 'rdtsc', 'rep', 'repe', 'repne', 'repnz', 'repz', 'ret', 'retf', 'retn', 'rol', 'ror', 'rsdc', 'rsldt', 'rsm', 'sahf', 'sal', 'salc', 'sar', 'sbb', 'scasb', 'scasd', 'scasw', 'seta', 'setae', 'setb', 'setbe', 'setc', 'setcxz', 'sete', 'setg', 'setge', 'setl', 'setle', 'setna', 'setnae', 'setnb', 'setnbe', 'setnc', 'setne', 'setng', 'setnge', 'setnl', 'setnle', 'setno', 'setnp', 'setns', 'setnz', 'seto', 'setp', 'setpe', 'setpo', 'sets', 'setz', 'sgdt', 'shl', 'shld', 'shr', 'shrd', 'sidt', 'sldt', 'smi', 'smint', 'smintold', 'smsw', 'stc', 'std', 'sti', 'stosb', 'stosd', 'stosw', 'str', 'sub', 'svdc', 'svldt', 'svts', 'syscall', 'sysenter', 'sysexit', 'sysret', 'test', 'ud1', 'ud2', 'umov', 'verr', 'verw', 'wait', 'wbinvd', 'wrmsr', 'wrshr', 'xadd', 'xbts', 'xchg', 'xlat', 'xlatb', 'xor'))

    def __init__(self, **options):
        Lexer.__init__(self, **options)
        self.keywords = set()
        if get_bool_opt(options, 'turbopascal', True):
            self.keywords.update(self.TURBO_PASCAL_KEYWORDS)
        if get_bool_opt(options, 'delphi', True):
            self.keywords.update(self.DELPHI_KEYWORDS)
        if get_bool_opt(options, 'freepascal', True):
            self.keywords.update(self.FREE_PASCAL_KEYWORDS)
        self.builtins = set()
        for unit in get_list_opt(options, 'units', list(self.BUILTIN_UNITS)):
            self.builtins.update(self.BUILTIN_UNITS[unit])

    def get_tokens_unprocessed(self, text):
        scanner = Scanner(text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        stack = ['initial']
        in_function_block = False
        in_property_block = False
        was_dot = False
        next_token_is_function = False
        next_token_is_property = False
        collect_labels = False
        block_labels = set()
        brace_balance = [0, 0]
        while not scanner.eos:
            token = Error
            if stack[-1] == 'initial':
                if scanner.scan('\\s+'):
                    token = Text
                elif scanner.scan('\\{.*?\\}|\\(\\*.*?\\*\\)'):
                    if scanner.match.startswith('$'):
                        token = Comment.Preproc
                    else:
                        token = Comment.Multiline
                elif scanner.scan('//.*?$'):
                    token = Comment.Single
                elif scanner.scan('[-+*\\/=<>:;,.@\\^]'):
                    token = Operator
                    if collect_labels and scanner.match == ';':
                        collect_labels = False
                elif scanner.scan('[\\(\\)\\[\\]]+'):
                    token = Punctuation
                    next_token_is_function = False
                    if in_function_block or in_property_block:
                        if scanner.match == '(':
                            brace_balance[0] += 1
                        elif scanner.match == ')':
                            brace_balance[0] -= 1
                        elif scanner.match == '[':
                            brace_balance[1] += 1
                        elif scanner.match == ']':
                            brace_balance[1] -= 1
                elif scanner.scan('[A-Za-z_][A-Za-z_0-9]*'):
                    lowercase_name = scanner.match.lower()
                    if lowercase_name == 'result':
                        token = Name.Builtin.Pseudo
                    elif lowercase_name in self.keywords:
                        token = Keyword
                        if (in_function_block or in_property_block) and lowercase_name in self.BLOCK_KEYWORDS and (brace_balance[0] <= 0) and (brace_balance[1] <= 0):
                            in_function_block = False
                            in_property_block = False
                            brace_balance = [0, 0]
                            block_labels = set()
                        if lowercase_name in ('label', 'goto'):
                            collect_labels = True
                        elif lowercase_name == 'asm':
                            stack.append('asm')
                        elif lowercase_name == 'property':
                            in_property_block = True
                            next_token_is_property = True
                        elif lowercase_name in ('procedure', 'operator', 'function', 'constructor', 'destructor'):
                            in_function_block = True
                            next_token_is_function = True
                    elif in_function_block and lowercase_name in self.FUNCTION_MODIFIERS:
                        token = Keyword.Pseudo
                    elif in_property_block and lowercase_name in ('read', 'write'):
                        token = Keyword.Pseudo
                        next_token_is_function = True
                    elif next_token_is_function:
                        if scanner.test('\\s*\\.\\s*'):
                            token = Name.Class
                        else:
                            token = Name.Function
                            next_token_is_function = False
                    elif next_token_is_property:
                        token = Name.Property
                        next_token_is_property = False
                    elif collect_labels:
                        token = Name.Label
                        block_labels.add(scanner.match.lower())
                    elif lowercase_name in block_labels:
                        token = Name.Label
                    elif lowercase_name in self.BUILTIN_TYPES:
                        token = Keyword.Type
                    elif lowercase_name in self.DIRECTIVES:
                        token = Keyword.Pseudo
                    elif not was_dot and lowercase_name in self.builtins:
                        token = Name.Builtin
                    else:
                        token = Name
                elif scanner.scan("'"):
                    token = String
                    stack.append('string')
                elif scanner.scan('\\#(\\d+|\\$[0-9A-Fa-f]+)'):
                    token = String.Char
                elif scanner.scan('\\$[0-9A-Fa-f]+'):
                    token = Number.Hex
                elif scanner.scan('\\d+(?![eE]|\\.[^.])'):
                    token = Number.Integer
                elif scanner.scan('\\d+(\\.\\d+([eE][+-]?\\d+)?|[eE][+-]?\\d+)'):
                    token = Number.Float
                else:
                    if len(stack) > 1:
                        stack.pop()
                    scanner.get_char()
            elif stack[-1] == 'string':
                if scanner.scan("''"):
                    token = String.Escape
                elif scanner.scan("'"):
                    token = String
                    stack.pop()
                elif scanner.scan("[^']*"):
                    token = String
                else:
                    scanner.get_char()
                    stack.pop()
            elif stack[-1] == 'asm':
                if scanner.scan('\\s+'):
                    token = Text
                elif scanner.scan('end'):
                    token = Keyword
                    stack.pop()
                elif scanner.scan('\\{.*?\\}|\\(\\*.*?\\*\\)'):
                    if scanner.match.startswith('$'):
                        token = Comment.Preproc
                    else:
                        token = Comment.Multiline
                elif scanner.scan('//.*?$'):
                    token = Comment.Single
                elif scanner.scan("'"):
                    token = String
                    stack.append('string')
                elif scanner.scan('@@[A-Za-z_][A-Za-z_0-9]*'):
                    token = Name.Label
                elif scanner.scan('[A-Za-z_][A-Za-z_0-9]*'):
                    lowercase_name = scanner.match.lower()
                    if lowercase_name in self.ASM_INSTRUCTIONS:
                        token = Keyword
                    elif lowercase_name in self.ASM_REGISTERS:
                        token = Name.Builtin
                    else:
                        token = Name
                elif scanner.scan('[-+*\\/=<>:;,.@\\^]+'):
                    token = Operator
                elif scanner.scan('[\\(\\)\\[\\]]+'):
                    token = Punctuation
                elif scanner.scan('\\$[0-9A-Fa-f]+'):
                    token = Number.Hex
                elif scanner.scan('\\d+(?![eE]|\\.[^.])'):
                    token = Number.Integer
                elif scanner.scan('\\d+(\\.\\d+([eE][+-]?\\d+)?|[eE][+-]?\\d+)'):
                    token = Number.Float
                else:
                    scanner.get_char()
                    stack.pop()
            if scanner.match.strip():
                was_dot = scanner.match == '.'
            yield (scanner.start_pos, token, scanner.match or '')