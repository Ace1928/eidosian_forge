import requests
class ContextAlreadyExists(DockerException):

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f'context {self.name} already exists'