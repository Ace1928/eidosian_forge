def __read_copyright_line(record, line):
    if line.startswith('+----'):
        return False
    return True