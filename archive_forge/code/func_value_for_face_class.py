from snappy.ptolemy.homology import homology_basis_representatives_with_orders
def value_for_face_class(weights, face_class):
    """
    Given weights per face per tetrahedron and a face_class as encoded
    by the ptolemy module, extract the weight for that face_class and
    perform consistency check.
    """
    sgn, power, repr0, repr1 = face_class
    val0 = weights[face_var_name_to_index(repr0)]
    val1 = weights[face_var_name_to_index(repr1)]
    if abs(val0 - sgn * val1) > 1e-06:
        raise ValueError('Weights for identified faces do not match')
    return val0