from pyparsing import ParserElement,LineEnd,Optional,Word,nums,Regex,\
class TAPTest(object):

    def __init__(self, results):
        self.num = results.testNumber
        self.passed = results.passed == 'ok'
        self.skipped = self.todo = False
        if results.directive:
            self.skipped = results.directive[0][0] == 'SKIP'
            self.todo = results.directive[0][0] == 'TODO'

    @classmethod
    def bailedTest(cls, num):
        ret = TAPTest(empty.parseString(''))
        ret.num = num
        ret.skipped = True
        return ret