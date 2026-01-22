class ParameterGroup(dict):

    def __init__(self, connection=None):
        dict.__init__(self)
        self.connection = connection
        self.name = None
        self.description = None
        self.engine = None
        self._current_param = None

    def __repr__(self):
        return 'ParameterGroup:%s' % self.name

    def startElement(self, name, attrs, connection):
        if name == 'Parameter':
            if self._current_param:
                self[self._current_param.name] = self._current_param
            self._current_param = Parameter(self)
            return self._current_param

    def endElement(self, name, value, connection):
        if name == 'DBParameterGroupName':
            self.name = value
        elif name == 'Description':
            self.description = value
        elif name == 'Engine':
            self.engine = value
        else:
            setattr(self, name, value)

    def modifiable(self):
        mod = []
        for key in self:
            p = self[key]
            if p.is_modifiable:
                mod.append(p)
        return mod

    def get_params(self):
        pg = self.connection.get_all_dbparameters(self.name)
        self.update(pg)

    def add_param(self, name, value, apply_method):
        param = Parameter()
        param.name = name
        param.value = value
        param.apply_method = apply_method
        self.params.append(param)