from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def are_parallel(u, v):
    """
    >>> a = Vector3([1, 0, 0])
    >>> b = Vector3([1, 1, 1])
    >>> are_parallel(a, b)
    False
    >>> are_parallel(a, 2*a)
    True
    """
    if u == 0 or v == 0:
        return True
    i = min_support(u)
    if v[i] == 0:
        return False
    return u / u[i] == v / v[i]