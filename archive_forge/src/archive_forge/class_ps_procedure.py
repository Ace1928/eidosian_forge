from fontTools.encodings.StandardEncoding import StandardEncoding
class ps_procedure(ps_object):
    literal = 0

    def __repr__(self):
        return '<procedure>'

    def __str__(self):
        psstring = '{'
        for i in range(len(self.value)):
            if i:
                psstring = psstring + ' ' + str(self.value[i])
            else:
                psstring = psstring + str(self.value[i])
        return psstring + '}'