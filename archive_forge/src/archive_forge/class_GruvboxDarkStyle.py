from pygments.style import Style
from pygments.token import Token, Keyword, Name, Comment, String, Error, \
class GruvboxDarkStyle(Style):
    """
    Pygments version of the "gruvbox" dark vim theme.
    """
    background_color = '#282828'
    highlight_color = '#ebdbb2'
    styles = {Token: '#dddddd', Comment: 'italic #928374', Comment.PreProc: '#8ec07c', Comment.Special: 'bold italic #ebdbb2', Keyword: '#fb4934', Operator.Word: '#fb4934', String: '#b8bb26', String.Escape: '#fe8019', Number: '#d3869b', Name.Builtin: '#fe8019', Name.Variable: '#83a598', Name.Constant: '#d3869b', Name.Class: '#8ec07c', Name.Function: '#8ec07c', Name.Namespace: '#8ec07c', Name.Exception: '#fb4934', Name.Tag: '#8ec07c', Name.Attribute: '#fabd2f', Name.Decorator: '#fb4934', Generic.Heading: 'bold #ebdbb2', Generic.Subheading: 'underline #ebdbb2', Generic.Deleted: 'bg:#fb4934 #282828', Generic.Inserted: 'bg:#b8bb26 #282828', Generic.Error: '#fb4934', Generic.Emph: 'italic', Generic.Strong: 'bold', Generic.EmphStrong: 'bold italic', Generic.Prompt: '#a89984', Generic.Output: '#f2e5bc', Generic.Traceback: '#fb4934', Error: 'bg:#fb4934 #282828'}