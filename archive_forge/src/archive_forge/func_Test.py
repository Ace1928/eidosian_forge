from pyparsing import Word, alphanums, Keyword, Group, Combine, Forward, Suppress, OneOrMore, oneOf
def Test(self):
    all_ok = True
    for item in list(self.tests.keys()):
        print(item)
        r = self.Parse(item)
        e = self.tests[item]
        print('Result: %s' % r)
        print('Expect: %s' % e)
        if e == r:
            print('Test OK')
        else:
            all_ok = False
            print('>>>>>>>>>>>>>>>>>>>>>>Test ERROR<<<<<<<<<<<<<<<<<<<<<')
        print('')
    return all_ok