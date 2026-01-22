class Qualifications(object):

    def __init__(self, requirements=None):
        if requirements is None:
            requirements = []
        self.requirements = requirements

    def add(self, req):
        self.requirements.append(req)

    def get_as_params(self):
        params = {}
        assert len(self.requirements) <= 10
        for n, req in enumerate(self.requirements):
            reqparams = req.get_as_params()
            for rp in reqparams:
                params['QualificationRequirement.%s.%s' % (n + 1, rp)] = reqparams[rp]
        return params