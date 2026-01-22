import re
def _patch_pow0_py35(pq):
    try:
        pq.metre ** 0
    except Exception:
        pq.quantity.Quantity.__pow__ = pq.quantity.check_uniform(lambda self, other: np.power(self, other))