import docutils.utils.math.tex2unichar as tex2unichar
def xml_end(self):
    return ['</%s>' % self.__class__.__name__]