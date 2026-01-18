import greenlet
def tracefunc(*args):
    print('TRACE', *args)
    global switch_to_g2
    if switch_to_g2:
        switch_to_g2 = False
        g2.switch()
    print('\tLEAVE TRACE', *args)