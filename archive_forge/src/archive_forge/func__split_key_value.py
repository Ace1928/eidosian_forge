def _split_key_value(self, line):
    colon = line.find(':')
    equal = line.find('=')
    if colon < 0 and equal < 0:
        return self.error_invalid_assignment(line)
    if colon < 0 or (equal >= 0 and equal < colon):
        key, value = (line[:equal], line[equal + 1:])
    else:
        key, value = (line[:colon], line[colon + 1:])
    value = value.strip()
    if value and value[0] == value[-1] and value.startswith(('"', "'")):
        value = value[1:-1]
    return (key.strip(), [value])