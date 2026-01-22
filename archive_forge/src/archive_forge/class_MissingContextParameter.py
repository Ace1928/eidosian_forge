import requests
class MissingContextParameter(DockerException):

    def __init__(self, param):
        self.param = param

    def __str__(self):
        return f'missing parameter: {self.param}'