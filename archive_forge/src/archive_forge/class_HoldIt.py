import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy import array
class HoldIt:

    def __init__(self, name):
        self.name = name

    def save(self, what=None, filename=None, header=True, useinstant=True, comment=None):
        if what is None:
            what = (i for i in self.__dict__ if i[0] != '_')
        if header:
            txt = ['import numpy as np\nfrom numpy import array\n\n']
            if useinstant:
                txt.append('from statsmodels.tools.testing import Holder\n\n')
        else:
            txt = []
        if useinstant:
            txt.append('%s = Holder()' % self.name)
            prefix = '%s.' % self.name
        else:
            prefix = ''
        if comment is not None:
            txt.append("{}comment = '{}'".format(prefix, comment))
        for x in what:
            txt.append('{}{} = {}'.format(prefix, x, repr(getattr(self, x))))
        txt.extend(['', ''])
        if filename is not None:
            with open(filename, 'a+', encoding='utf-8') as fd:
                fd.write('\n'.join(txt))
        return txt