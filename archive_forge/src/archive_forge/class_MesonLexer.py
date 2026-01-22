from pygments.lexer import RegexLexer, words, include
from pygments.token import Comment, Name, Number, Punctuation, Operator, \
class MesonLexer(RegexLexer):
    """Meson language lexer.

    The grammar definition use to transcribe the syntax was retrieved from
    https://mesonbuild.com/Syntax.html#grammar for version 0.58.
    Some of those definitions are improperly transcribed, so the Meson++
    implementation was also checked: https://github.com/dcbaker/meson-plus-plus.

    .. versionadded:: 2.10
    """
    name = 'Meson'
    url = 'https://mesonbuild.com/'
    aliases = ['meson', 'meson.build']
    filenames = ['meson.build', 'meson_options.txt']
    mimetypes = ['text/x-meson']
    tokens = {'root': [('#.*?$', Comment), ("'''.*'''", String.Single), ('[1-9][0-9]*', Number.Integer), ('0o[0-7]+', Number.Oct), ('0x[a-fA-F0-9]+', Number.Hex), include('string'), include('keywords'), include('expr'), ('[a-zA-Z_][a-zA-Z_0-9]*', Name), ('\\s+', Whitespace)], 'string': [("[']{3}([']{0,2}([^\\\\']|\\\\(.|\\n)))*[']{3}", String), ("'.*?(?<!\\\\)(\\\\\\\\)*?'", String)], 'keywords': [(words(('if', 'elif', 'else', 'endif', 'foreach', 'endforeach', 'break', 'continue'), suffix='\\b'), Keyword)], 'expr': [('(in|and|or|not)\\b', Operator.Word), ('(\\*=|/=|%=|\\+]=|-=|==|!=|\\+|-|=)', Operator), ('[\\[\\]{}:().,?]', Punctuation), (words(('true', 'false'), suffix='\\b'), Keyword.Constant), include('builtins'), (words(('meson', 'build_machine', 'host_machine', 'target_machine'), suffix='\\b'), Name.Variable.Magic)], 'builtins': [(words(('add_global_arguments', 'add_global_link_arguments', 'add_languages', 'add_project_arguments', 'add_project_link_arguments', 'add_test_setup', 'assert', 'benchmark', 'both_libraries', 'build_target', 'configuration_data', 'configure_file', 'custom_target', 'declare_dependency', 'dependency', 'disabler', 'environment', 'error', 'executable', 'files', 'find_library', 'find_program', 'generator', 'get_option', 'get_variable', 'include_directories', 'install_data', 'install_headers', 'install_man', 'install_subdir', 'is_disabler', 'is_variable', 'jar', 'join_paths', 'library', 'message', 'project', 'range', 'run_command', 'set_variable', 'shared_library', 'shared_module', 'static_library', 'subdir', 'subdir_done', 'subproject', 'summary', 'test', 'vcs_tag', 'warning'), prefix='(?<!\\.)', suffix='\\b'), Name.Builtin), ('(?<!\\.)import\\b', Name.Namespace)]}