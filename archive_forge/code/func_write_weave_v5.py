def write_weave_v5(weave, f):
    """Write weave to file f."""
    f.write(FORMAT_1)
    for version, included in enumerate(weave._parents):
        if included:
            mininc = included
            f.write(b'i ')
            f.write(b' '.join((b'%d' % i for i in mininc)))
            f.write(b'\n')
        else:
            f.write(b'i\n')
        f.write(b'1 ' + weave._sha1s[version] + b'\n')
        f.write(b'n ' + weave._names[version] + b'\n')
        f.write(b'\n')
    f.write(b'w\n')
    for l in weave._weave:
        if isinstance(l, tuple):
            if l[0] == b'}':
                f.write(b'}\n')
            else:
                f.write(l[0] + b' %d\n' % l[1])
        elif not l:
            f.write(b', \n')
        elif l.endswith(b'\n'):
            f.write(b'. ' + l)
        else:
            f.write(b', ' + l + b'\n')
    f.write(b'W\n')