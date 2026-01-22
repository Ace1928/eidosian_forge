def _makeunicodes(f):
    lines = iter(f.readlines())
    unicodes = {}
    for line in lines:
        if not line:
            continue
        num, name = line.split(';')[:2]
        if name[0] == '<':
            continue
        num = int(num, 16)
        unicodes[num] = name
    return unicodes