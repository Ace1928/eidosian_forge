def _merge_lists(b, c):
    count = 0
    i = j = 0
    a = []
    while i < len(b) and j < len(c):
        count += 1
        if b[i] <= c[j]:
            a.append(b[i])
            i += 1
        else:
            a.append(c[j])
            j += 1
    if i == len(b):
        a += c[j:]
    else:
        a += b[i:]
    return (a, count)