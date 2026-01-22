import warnings
def _create_opt_rule(self, rulename):
    """ Given a rule name, creates an optional ply.yacc rule
            for it. The name of the optional rule is
            <rulename>_opt
        """
    optname = rulename + '_opt'

    def optrule(self, p):
        p[0] = p[1]
    optrule.__doc__ = '%s : empty\n| %s' % (optname, rulename)
    optrule.__name__ = 'p_%s' % optname
    setattr(self.__class__, optrule.__name__, optrule)