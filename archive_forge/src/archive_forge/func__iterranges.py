def _iterranges(self, r1, r2, minval=_MININF, maxval=_MAXINF):
    curval = minval
    curstates = {'r1': False, 'r2': False}
    imax, jmax = (2 * len(r1), 2 * len(r2))
    i, j = (0, 0)
    while i < imax or j < jmax:
        if i < imax and (j < jmax and r1[i >> 1][i & 1] < r2[j >> 1][j & 1] or j == jmax):
            cur_r, newname, newstate = (r1[i >> 1][i & 1], 'r1', not i & 1)
            i += 1
        else:
            cur_r, newname, newstate = (r2[j >> 1][j & 1], 'r2', not j & 1)
            j += 1
        if curval < cur_r:
            if cur_r > maxval:
                break
            yield (curstates, (curval, cur_r))
            curval = cur_r
        curstates[newname] = newstate
    if curval < maxval:
        yield (curstates, (curval, maxval))