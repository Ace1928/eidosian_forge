class ParamikoNotPresent(DependencyNotPresent):
    _fmt = 'Unable to import paramiko (required for sftp support): %(error)s'

    def __init__(self, error):
        DependencyNotPresent.__init__(self, 'paramiko', error)