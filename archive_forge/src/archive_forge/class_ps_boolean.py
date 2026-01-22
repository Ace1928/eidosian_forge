from fontTools.encodings.StandardEncoding import StandardEncoding
class ps_boolean(ps_object):

    def __str__(self):
        if self.value:
            return 'true'
        else:
            return 'false'