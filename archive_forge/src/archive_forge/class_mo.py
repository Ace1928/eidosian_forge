import docutils.utils.math.tex2unichar as tex2unichar
class mo(mx):
    translation = {'<': '&lt;', '>': '&gt;'}

    def xml_body(self):
        return [self.translation.get(self.data, self.data)]