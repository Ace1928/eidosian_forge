def __read_reference_line(record, line):
    if not line.strip():
        return False
    reference = record.references[-1]
    if line.startswith('     '):
        if reference.authors[-1] == ',':
            reference.authors += line[4:].rstrip()
        else:
            reference.citation += line[5:]
        return True
    raise Exception(f"I don't understand the reference line\n{line}")