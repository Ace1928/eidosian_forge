import enum
class ResultsFormat(str, enum.Enum):
    osrl = 'osrl'
    results = 'results'
    sol = 'sol'
    soln = 'soln'
    yaml = 'yaml'
    json = 'json'

    def __str__(self):
        return self.value