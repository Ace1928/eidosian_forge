class InvalidSchemeHeaders(ParseException):

    def __str__(self):
        return 'Contradictory scheme headers'