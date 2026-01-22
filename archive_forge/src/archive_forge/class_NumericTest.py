import datetime
import decimal
import json
import re
import uuid
from .. import config
from .. import engines
from .. import fixtures
from .. import mock
from ..assertions import eq_
from ..assertions import is_
from ..assertions import ne_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import and_
from ... import ARRAY
from ... import BigInteger
from ... import bindparam
from ... import Boolean
from ... import case
from ... import cast
from ... import Date
from ... import DateTime
from ... import Float
from ... import Integer
from ... import Interval
from ... import JSON
from ... import literal
from ... import literal_column
from ... import MetaData
from ... import null
from ... import Numeric
from ... import select
from ... import String
from ... import testing
from ... import Text
from ... import Time
from ... import TIMESTAMP
from ... import type_coerce
from ... import TypeDecorator
from ... import Unicode
from ... import UnicodeText
from ... import UUID
from ... import Uuid
from ...orm import declarative_base
from ...orm import Session
from ...sql import sqltypes
from ...sql.sqltypes import LargeBinary
from ...sql.sqltypes import PickleType
class NumericTest(_LiteralRoundTripFixture, fixtures.TestBase):
    __backend__ = True

    @testing.fixture
    def do_numeric_test(self, metadata, connection):

        def run(type_, input_, output, filter_=None, check_scale=False):
            t = Table('t', metadata, Column('x', type_))
            t.create(connection)
            connection.execute(t.insert(), [{'x': x} for x in input_])
            result = {row[0] for row in connection.execute(t.select())}
            output = set(output)
            if filter_:
                result = {filter_(x) for x in result}
                output = {filter_(x) for x in output}
            eq_(result, output)
            if check_scale:
                eq_([str(x) for x in result], [str(x) for x in output])
            connection.execute(t.delete())
            if type_.asdecimal:
                test_value = decimal.Decimal('2.9')
                add_value = decimal.Decimal('37.12')
            else:
                test_value = 2.9
                add_value = 37.12
            connection.execute(t.insert(), {'x': test_value})
            assert_we_are_a_number = connection.scalar(select(type_coerce(t.c.x + add_value, type_)))
            eq_(round(assert_we_are_a_number, 3), round(test_value + add_value, 3))
        return run

    def test_render_literal_numeric(self, literal_round_trip):
        literal_round_trip(Numeric(precision=8, scale=4), [15.7563, decimal.Decimal('15.7563')], [decimal.Decimal('15.7563')])

    def test_render_literal_numeric_asfloat(self, literal_round_trip):
        literal_round_trip(Numeric(precision=8, scale=4, asdecimal=False), [15.7563, decimal.Decimal('15.7563')], [15.7563])

    def test_render_literal_float(self, literal_round_trip):
        literal_round_trip(Float(), [15.7563, decimal.Decimal('15.7563')], [15.7563], filter_=lambda n: n is not None and round(n, 5) or None, support_whereclause=False)

    @testing.requires.precision_generic_float_type
    def test_float_custom_scale(self, do_numeric_test):
        do_numeric_test(Float(None, decimal_return_scale=7, asdecimal=True), [15.7563827, decimal.Decimal('15.7563827')], [decimal.Decimal('15.7563827')], check_scale=True)

    def test_numeric_as_decimal(self, do_numeric_test):
        do_numeric_test(Numeric(precision=8, scale=4), [15.7563, decimal.Decimal('15.7563')], [decimal.Decimal('15.7563')])

    def test_numeric_as_float(self, do_numeric_test):
        do_numeric_test(Numeric(precision=8, scale=4, asdecimal=False), [15.7563, decimal.Decimal('15.7563')], [15.7563])

    @testing.requires.infinity_floats
    def test_infinity_floats(self, do_numeric_test):
        """test for #977, #7283"""
        do_numeric_test(Float(None), [float('inf')], [float('inf')])

    @testing.requires.fetch_null_from_numeric
    def test_numeric_null_as_decimal(self, do_numeric_test):
        do_numeric_test(Numeric(precision=8, scale=4), [None], [None])

    @testing.requires.fetch_null_from_numeric
    def test_numeric_null_as_float(self, do_numeric_test):
        do_numeric_test(Numeric(precision=8, scale=4, asdecimal=False), [None], [None])

    @testing.requires.floats_to_four_decimals
    def test_float_as_decimal(self, do_numeric_test):
        do_numeric_test(Float(asdecimal=True), [15.756, decimal.Decimal('15.756'), None], [decimal.Decimal('15.756'), None], filter_=lambda n: n is not None and round(n, 4) or None)

    def test_float_as_float(self, do_numeric_test):
        do_numeric_test(Float(), [15.756, decimal.Decimal('15.756')], [15.756], filter_=lambda n: n is not None and round(n, 5) or None)

    @testing.requires.literal_float_coercion
    def test_float_coerce_round_trip(self, connection):
        expr = 15.7563
        val = connection.scalar(select(literal(expr)))
        eq_(val, expr)

    @testing.requires.implicit_decimal_binds
    def test_decimal_coerce_round_trip(self, connection):
        expr = decimal.Decimal('15.7563')
        val = connection.scalar(select(literal(expr)))
        eq_(val, expr)

    def test_decimal_coerce_round_trip_w_cast(self, connection):
        expr = decimal.Decimal('15.7563')
        val = connection.scalar(select(cast(expr, Numeric(10, 4))))
        eq_(val, expr)

    @testing.requires.precision_numerics_general
    def test_precision_decimal(self, do_numeric_test):
        numbers = {decimal.Decimal('54.234246451650'), decimal.Decimal('0.004354'), decimal.Decimal('900.0')}
        do_numeric_test(Numeric(precision=18, scale=12), numbers, numbers)

    @testing.requires.precision_numerics_enotation_large
    def test_enotation_decimal(self, do_numeric_test):
        """test exceedingly small decimals.

        Decimal reports values with E notation when the exponent
        is greater than 6.

        """
        numbers = {decimal.Decimal('1E-2'), decimal.Decimal('1E-3'), decimal.Decimal('1E-4'), decimal.Decimal('1E-5'), decimal.Decimal('1E-6'), decimal.Decimal('1E-7'), decimal.Decimal('1E-8'), decimal.Decimal('0.01000005940696'), decimal.Decimal('0.00000005940696'), decimal.Decimal('0.00000000000696'), decimal.Decimal('0.70000000000696'), decimal.Decimal('696E-12')}
        do_numeric_test(Numeric(precision=18, scale=14), numbers, numbers)

    @testing.requires.precision_numerics_enotation_large
    def test_enotation_decimal_large(self, do_numeric_test):
        """test exceedingly large decimals."""
        numbers = {decimal.Decimal('4E+8'), decimal.Decimal('5748E+15'), decimal.Decimal('1.521E+15'), decimal.Decimal('00000000000000.1E+12')}
        do_numeric_test(Numeric(precision=25, scale=2), numbers, numbers)

    @testing.requires.precision_numerics_many_significant_digits
    def test_many_significant_digits(self, do_numeric_test):
        numbers = {decimal.Decimal('31943874831932418390.01'), decimal.Decimal('319438950232418390.273596'), decimal.Decimal('87673.594069654243')}
        do_numeric_test(Numeric(precision=38, scale=12), numbers, numbers)

    @testing.requires.precision_numerics_retains_significant_digits
    def test_numeric_no_decimal(self, do_numeric_test):
        numbers = {decimal.Decimal('1.000')}
        do_numeric_test(Numeric(precision=5, scale=3), numbers, numbers, check_scale=True)

    @testing.combinations(sqltypes.Float, sqltypes.Double, argnames='cls_')
    @testing.requires.float_is_numeric
    def test_float_is_not_numeric(self, connection, cls_):
        target_type = cls_().dialect_impl(connection.dialect)
        numeric_type = sqltypes.Numeric().dialect_impl(connection.dialect)
        ne_(target_type.__visit_name__, numeric_type.__visit_name__)
        ne_(target_type.__class__, numeric_type.__class__)