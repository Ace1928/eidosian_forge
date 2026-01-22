import re
from pygments.lexer import Lexer, RegexLexer, bygroups, words, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class ErlangLexer(RegexLexer):
    """
    For the Erlang functional programming language.

    Blame Jeremy Thurgood (http://jerith.za.net/).

    .. versionadded:: 0.9
    """
    name = 'Erlang'
    aliases = ['erlang']
    filenames = ['*.erl', '*.hrl', '*.es', '*.escript']
    mimetypes = ['text/x-erlang']
    keywords = ('after', 'begin', 'case', 'catch', 'cond', 'end', 'fun', 'if', 'let', 'of', 'query', 'receive', 'try', 'when')
    builtins = ('abs', 'append_element', 'apply', 'atom_to_list', 'binary_to_list', 'bitstring_to_list', 'binary_to_term', 'bit_size', 'bump_reductions', 'byte_size', 'cancel_timer', 'check_process_code', 'delete_module', 'demonitor', 'disconnect_node', 'display', 'element', 'erase', 'exit', 'float', 'float_to_list', 'fun_info', 'fun_to_list', 'function_exported', 'garbage_collect', 'get', 'get_keys', 'group_leader', 'hash', 'hd', 'integer_to_list', 'iolist_to_binary', 'iolist_size', 'is_atom', 'is_binary', 'is_bitstring', 'is_boolean', 'is_builtin', 'is_float', 'is_function', 'is_integer', 'is_list', 'is_number', 'is_pid', 'is_port', 'is_process_alive', 'is_record', 'is_reference', 'is_tuple', 'length', 'link', 'list_to_atom', 'list_to_binary', 'list_to_bitstring', 'list_to_existing_atom', 'list_to_float', 'list_to_integer', 'list_to_pid', 'list_to_tuple', 'load_module', 'localtime_to_universaltime', 'make_tuple', 'md5', 'md5_final', 'md5_update', 'memory', 'module_loaded', 'monitor', 'monitor_node', 'node', 'nodes', 'open_port', 'phash', 'phash2', 'pid_to_list', 'port_close', 'port_command', 'port_connect', 'port_control', 'port_call', 'port_info', 'port_to_list', 'process_display', 'process_flag', 'process_info', 'purge_module', 'put', 'read_timer', 'ref_to_list', 'register', 'resume_process', 'round', 'send', 'send_after', 'send_nosuspend', 'set_cookie', 'setelement', 'size', 'spawn', 'spawn_link', 'spawn_monitor', 'spawn_opt', 'split_binary', 'start_timer', 'statistics', 'suspend_process', 'system_flag', 'system_info', 'system_monitor', 'system_profile', 'term_to_binary', 'tl', 'trace', 'trace_delivered', 'trace_info', 'trace_pattern', 'trunc', 'tuple_size', 'tuple_to_list', 'universaltime_to_localtime', 'unlink', 'unregister', 'whereis')
    operators = '(\\+\\+?|--?|\\*|/|<|>|/=|=:=|=/=|=<|>=|==?|<-|!|\\?)'
    word_operators = ('and', 'andalso', 'band', 'bnot', 'bor', 'bsl', 'bsr', 'bxor', 'div', 'not', 'or', 'orelse', 'rem', 'xor')
    atom_re = "(?:[a-z]\\w*|'[^\\n']*[^\\\\]')"
    variable_re = '(?:[A-Z_]\\w*)'
    esc_char_re = '[bdefnrstv\\\'"\\\\]'
    esc_octal_re = '[0-7][0-7]?[0-7]?'
    esc_hex_re = '(?:x[0-9a-fA-F]{2}|x\\{[0-9a-fA-F]+\\})'
    esc_ctrl_re = '\\^[a-zA-Z]'
    escape_re = '(?:\\\\(?:' + esc_char_re + '|' + esc_octal_re + '|' + esc_hex_re + '|' + esc_ctrl_re + '))'
    macro_re = '(?:' + variable_re + '|' + atom_re + ')'
    base_re = '(?:[2-9]|[12][0-9]|3[0-6])'
    tokens = {'root': [('\\s+', Text), ('%.*\\n', Comment), (words(keywords, suffix='\\b'), Keyword), (words(builtins, suffix='\\b'), Name.Builtin), (words(word_operators, suffix='\\b'), Operator.Word), ('^-', Punctuation, 'directive'), (operators, Operator), ('"', String, 'string'), ('<<', Name.Label), ('>>', Name.Label), ('(' + atom_re + ')(:)', bygroups(Name.Namespace, Punctuation)), ('(?:^|(?<=:))(' + atom_re + ')(\\s*)(\\()', bygroups(Name.Function, Text, Punctuation)), ('[+-]?' + base_re + '#[0-9a-zA-Z]+', Number.Integer), ('[+-]?\\d+', Number.Integer), ('[+-]?\\d+.\\d+', Number.Float), ('[]\\[:_@\\".{}()|;,]', Punctuation), (variable_re, Name.Variable), (atom_re, Name), ('\\?' + macro_re, Name.Constant), ('\\$(?:' + escape_re + '|\\\\[ %]|[^\\\\])', String.Char), ('#' + atom_re + '(:?\\.' + atom_re + ')?', Name.Label), ('\\A#!.+\\n', Comment.Hashbang), ('#\\{', Punctuation, 'map_key')], 'string': [(escape_re, String.Escape), ('"', String, '#pop'), ('~[0-9.*]*[~#+BPWXb-ginpswx]', String.Interpol), ('[^"\\\\~]+', String), ('~', String)], 'directive': [('(define)(\\s*)(\\()(' + macro_re + ')', bygroups(Name.Entity, Text, Punctuation, Name.Constant), '#pop'), ('(record)(\\s*)(\\()(' + macro_re + ')', bygroups(Name.Entity, Text, Punctuation, Name.Label), '#pop'), (atom_re, Name.Entity, '#pop')], 'map_key': [include('root'), ('=>', Punctuation, 'map_val'), (':=', Punctuation, 'map_val'), ('\\}', Punctuation, '#pop')], 'map_val': [include('root'), (',', Punctuation, '#pop'), ('(?=\\})', Punctuation, '#pop')]}