def CompactListRepr(lst):
    """

  >>> CompactListRepr([0,1,1,1,1,0])
  '[0]+[1]*4+[0]'
  >>> CompactListRepr([0,1,1,2,1,1])
  '[0]+[1]*2+[2]+[1]*2'
  >>> CompactListRepr([])
  '[]'
  >>> CompactListRepr((0,1,1,1,1))
  '[0]+[1]*4'
  >>> CompactListRepr('foo')
  "['f']+['o']*2"

  """
    if not len(lst):
        return '[]'
    components = []
    last = lst[0]
    count = 1
    i = 1
    while i < len(lst):
        if lst[i] != last:
            label = '[%s]' % repr(last)
            if count > 1:
                label += '*%d' % count
            components.append(label)
            count = 1
            last = lst[i]
        else:
            count += 1
        i += 1
    if count != 0:
        label = '[%s]' % repr(last)
        if count > 1:
            label += '*%d' % count
        components.append(label)
    return '+'.join(components)