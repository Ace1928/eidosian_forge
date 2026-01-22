import configparser
def __remove_quotes(self, value):
    quotes = ["'", '"']
    for quote in quotes:
        if len(value) >= 2 and value[0] == value[-1] == quote:
            return value[1:-1]
    return value