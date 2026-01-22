import pygments
import pygments.formatter
class OdtPygmentsProgFormatter(OdtPygmentsFormatter):

    def format(self, tokensource, outfile):
        tokenclass = pygments.token.Token
        for ttype, value in tokensource:
            value = self.escape_function(value)
            if ttype == tokenclass.Keyword:
                s2 = self.rststyle('codeblock-keyword')
                s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            elif ttype == tokenclass.Literal.String:
                s2 = self.rststyle('codeblock-string')
                s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            elif ttype in (tokenclass.Literal.Number.Integer, tokenclass.Literal.Number.Integer.Long, tokenclass.Literal.Number.Float, tokenclass.Literal.Number.Hex, tokenclass.Literal.Number.Oct, tokenclass.Literal.Number):
                s2 = self.rststyle('codeblock-number')
                s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            elif ttype == tokenclass.Operator:
                s2 = self.rststyle('codeblock-operator')
                s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            elif ttype == tokenclass.Comment:
                s2 = self.rststyle('codeblock-comment')
                s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            elif ttype == tokenclass.Name.Class:
                s2 = self.rststyle('codeblock-classname')
                s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            elif ttype == tokenclass.Name.Function:
                s2 = self.rststyle('codeblock-functionname')
                s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            elif ttype == tokenclass.Name:
                s2 = self.rststyle('codeblock-name')
                s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            else:
                s1 = value
            outfile.write(s1)