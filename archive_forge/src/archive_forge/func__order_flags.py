def _order_flags(table):
    order = list(table.items())
    order.sort()
    order.reverse()
    return order