from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, \
class BlackWhiteStyle(Style):
    background_color = '#ffffff'
    default_style = ''
    styles = {Comment: 'italic', Comment.Preproc: 'noitalic', Keyword: 'bold', Keyword.Pseudo: 'nobold', Keyword.Type: 'nobold', Operator.Word: 'bold', Name.Class: 'bold', Name.Namespace: 'bold', Name.Exception: 'bold', Name.Entity: 'bold', Name.Tag: 'bold', String: 'italic', String.Interpol: 'bold', String.Escape: 'bold', Generic.Heading: 'bold', Generic.Subheading: 'bold', Generic.Emph: 'italic', Generic.Strong: 'bold', Generic.Prompt: 'bold', Error: 'border:#FF0000'}