from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, Operator
class AlgolStyle(Style):
    background_color = '#ffffff'
    default_style = ''
    styles = {Comment: 'italic #888', Comment.Preproc: 'bold noitalic #888', Comment.Special: 'bold noitalic #888', Keyword: 'underline bold', Keyword.Declaration: 'italic', Name.Builtin: 'bold italic', Name.Builtin.Pseudo: 'bold italic', Name.Namespace: 'bold italic #666', Name.Class: 'bold italic #666', Name.Function: 'bold italic #666', Name.Variable: 'bold italic #666', Name.Constant: 'bold italic #666', Operator.Word: 'bold', String: 'italic #666', Error: 'border:#FF0000'}