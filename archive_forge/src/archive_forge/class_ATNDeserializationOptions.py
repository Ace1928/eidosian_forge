class ATNDeserializationOptions(object):
    __slots__ = ('readonly', 'verifyATN', 'generateRuleBypassTransitions')
    defaultOptions = None

    def __init__(self, copyFrom: ATNDeserializationOptions=None):
        self.readonly = False
        self.verifyATN = True if copyFrom is None else copyFrom.verifyATN
        self.generateRuleBypassTransitions = False if copyFrom is None else copyFrom.generateRuleBypassTransitions

    def __setattr__(self, key, value):
        if key != 'readonly' and self.readonly:
            raise Exception('The object is read only.')
        super(type(self), self).__setattr__(key, value)