from pygments.lexer import RegexLexer, words, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, \
class GLShaderLexer(RegexLexer):
    """
    GLSL (OpenGL Shader) lexer.

    .. versionadded:: 1.1
    """
    name = 'GLSL'
    aliases = ['glsl']
    filenames = ['*.vert', '*.frag', '*.geo']
    mimetypes = ['text/x-glslsrc']
    tokens = {'root': [('^#.*', Comment.Preproc), ('//.*', Comment.Single), ('/(\\\\\\n)?[*](.|\\n)*?[*](\\\\\\n)?/', Comment.Multiline), ('\\+|-|~|!=?|\\*|/|%|<<|>>|<=?|>=?|==?|&&?|\\^|\\|\\|?', Operator), ('[?:]', Operator), ('\\bdefined\\b', Operator), ('[;{}(),\\[\\]]', Punctuation), ('[+-]?\\d*\\.\\d+([eE][-+]?\\d+)?', Number.Float), ('[+-]?\\d+\\.\\d*([eE][-+]?\\d+)?', Number.Float), ('0[xX][0-9a-fA-F]*', Number.Hex), ('0[0-7]*', Number.Oct), ('[1-9][0-9]*', Number.Integer), (words(('attribute', 'const', 'uniform', 'varying', 'centroid', 'break', 'continue', 'do', 'for', 'while', 'if', 'else', 'in', 'out', 'inout', 'float', 'int', 'void', 'bool', 'true', 'false', 'invariant', 'discard', 'return', 'mat2', 'mat3mat4', 'mat2x2', 'mat3x2', 'mat4x2', 'mat2x3', 'mat3x3', 'mat4x3', 'mat2x4', 'mat3x4', 'mat4x4', 'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4', 'bvec2', 'bvec3', 'bvec4', 'sampler1D', 'sampler2D', 'sampler3DsamplerCube', 'sampler1DShadow', 'sampler2DShadow', 'struct'), prefix='\\b', suffix='\\b'), Keyword), (words(('asm', 'class', 'union', 'enum', 'typedef', 'template', 'this', 'packed', 'goto', 'switch', 'default', 'inline', 'noinline', 'volatile', 'public', 'static', 'extern', 'external', 'interface', 'long', 'short', 'double', 'half', 'fixed', 'unsigned', 'lowp', 'mediump', 'highp', 'precision', 'input', 'output', 'hvec2', 'hvec3', 'hvec4', 'dvec2', 'dvec3', 'dvec4', 'fvec2', 'fvec3', 'fvec4', 'sampler2DRect', 'sampler3DRect', 'sampler2DRectShadow', 'sizeof', 'cast', 'namespace', 'using'), prefix='\\b', suffix='\\b'), Keyword), ('[a-zA-Z_]\\w*', Name), ('\\.', Punctuation), ('\\s+', Text)]}