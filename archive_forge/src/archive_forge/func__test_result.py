from ..sage_helper import _within_sage, sage_method, SageNotAvailable
def _test_result(number_field, prec=53, epsilon=1e-10):
    """
        sage: CF = ComplexField()
        sage: x = var('x')
        sage: nf = NumberField(x**2 + 1, 'x', embedding = CF(1.0j))
        sage: _test_result(nf)
        sage: nf = NumberField(x**2 + 7, 'x', embedding = CF(2.64575j))
        sage: _test_result(nf)
        sage: nf = NumberField(x**8 + 6 * x ** 4 + x + 23, 'x', embedding = CF(0.7747 + 1.25937j))
        sage: _test_result(nf, 212, epsilon = 1e-30)
    """
    CIF = ComplexIntervalField(prec)
    RIF = RealIntervalField(prec)
    epsilon = RIF(epsilon)
    real_number_field, x_expression, y_expression = field_containing_real_and_imaginary_part_of_number_field(number_field)
    x_val = x_expression.lift()(RIF(real_number_field.gen_embedding()))
    y_val = y_expression.lift()(RIF(real_number_field.gen_embedding()))
    z_val = CIF(x_val, y_val)
    diff = z_val - CIF(number_field.gen_embedding())
    if not abs(diff) < epsilon:
        raise Exception('Test failed')