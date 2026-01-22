from boto.exception import JSONResponseError
class DBSubnetGroupDoesNotCoverEnoughAZs(JSONResponseError):
    pass