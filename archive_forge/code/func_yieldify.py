def yieldify(rl):
    """ Turn a rule into a branching rule """

    def brl(expr):
        yield rl(expr)
    return brl