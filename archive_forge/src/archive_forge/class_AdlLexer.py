from pygments.lexer import RegexLexer, include, bygroups, using, default
from pygments.token import Text, Comment, Name, Literal, Number, String, \
class AdlLexer(AtomsLexer):
    """
    Lexer for ADL syntax.

    .. versionadded:: 2.1
    """
    name = 'ADL'
    aliases = ['adl']
    filenames = ['*.adl', '*.adls', '*.adlf', '*.adlx']
    tokens = {'whitespace': [('\\s*\\n', Text), ('^[ \\t]*--.*$', Comment)], 'odin_section': [('^(language|description|ontology|terminology|annotations|component_terminologies|revision_history)[ \\t]*\\n', Generic.Heading), ('^(definition)[ \\t]*\\n', Generic.Heading, 'cadl_section'), ('^([ \\t]*|[ \\t]+.*)\\n', using(OdinLexer)), ('^([^"]*")(>[ \\t]*\\n)', bygroups(String, Punctuation)), ('^----------*\\n', Text, '#pop'), ('^.*\\n', String), default('#pop')], 'cadl_section': [('^([ \\t]*|[ \\t]+.*)\\n', using(CadlLexer)), default('#pop')], 'rules_section': [('^[ \\t]+.*\\n', using(CadlLexer)), default('#pop')], 'metadata': [('\\)', Punctuation, '#pop'), (';', Punctuation), ('([Tt]rue|[Ff]alse)', Literal), ('\\d+(\\.\\d+)*', Literal), ('(\\d|[a-fA-F])+(-(\\d|[a-fA-F])+){3,}', Literal), ('\\w+', Name.Class), ('"', String, 'string'), ('=', Operator), ('[ \\t]+', Text), default('#pop')], 'root': [('^(archetype|template_overlay|operational_template|template|speciali[sz]e)', Generic.Heading), ('^(language|description|ontology|terminology|annotations|component_terminologies|revision_history)[ \\t]*\\n', Generic.Heading, 'odin_section'), ('^(definition)[ \\t]*\\n', Generic.Heading, 'cadl_section'), ('^(rules)[ \\t]*\\n', Generic.Heading, 'rules_section'), include('archetype_id'), ('[ \\t]*\\(', Punctuation, 'metadata'), include('whitespace')]}