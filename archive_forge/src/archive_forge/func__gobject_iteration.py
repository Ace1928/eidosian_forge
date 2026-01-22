def _gobject_iteration(*largs):
    loop = 0
    while context.pending() and loop < 10:
        context.iteration(False)
        loop += 1