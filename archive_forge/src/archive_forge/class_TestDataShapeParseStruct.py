from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
class TestDataShapeParseStruct(unittest.TestCase):

    def setUp(self):
        self.sym = datashape.TypeSymbolTable()

    def test_struct(self):
        self.assertEqual(parse('{x: int16, y: int32}', self.sym), ct.DataShape(ct.Record([('x', ct.DataShape(ct.int16)), ('y', ct.DataShape(ct.int32))])))
        self.assertEqual(parse('{x: int16, y: int32,}', self.sym), ct.DataShape(ct.Record([('x', ct.DataShape(ct.int16)), ('y', ct.DataShape(ct.int32))])))
        self.assertEqual(parse('{_x: int16, Zed: int32,}', self.sym), ct.DataShape(ct.Record([('_x', ct.DataShape(ct.int16)), ('Zed', ct.DataShape(ct.int32))])))
        ds_str = '3 * var * {\n                        id : int32,\n                        name : string,\n                        description : {\n                            language : string,\n                            text : string\n                        },\n                        entries : var * {\n                            date : date,\n                            text : string\n                        }\n                    }'
        int32 = ct.DataShape(ct.int32)
        string = ct.DataShape(ct.string)
        date = ct.DataShape(ct.date_)
        ds = (ct.Fixed(3), ct.Var(), ct.Record([('id', int32), ('name', string), ('description', ct.DataShape(ct.Record([('language', string), ('text', string)]))), ('entries', ct.DataShape(ct.Var(), ct.Record([('date', date), ('text', string)])))]))
        self.assertEqual(parse(ds_str, self.sym), ct.DataShape(*ds))

    def test_fields_with_dshape_names(self):
        ds = parse('{\n                type: bool,\n                data: bool,\n                blob: bool,\n                bool: bool,\n                int: int32,\n                float: float32,\n                double: float64,\n                int8: int8,\n                int16: int16,\n                int32: int32,\n                int64: int64,\n                uint8: uint8,\n                uint16: uint16,\n                uint32: uint32,\n                uint64: uint64,\n                float16: float32,\n                float32: float32,\n                float64: float64,\n                float128: float64,\n                complex: float32,\n                complex64: float32,\n                complex128: float64,\n                string: string,\n                object: string,\n                datetime: string,\n                datetime64: string,\n                timedelta: string,\n                timedelta64: string,\n                json: string,\n                var: string,\n            }', self.sym)
        self.assertEqual(type(ds[-1]), ct.Record)
        self.assertEqual(len(ds[-1].names), 30)

    def test_kiva_datashape(self):
        ds = parse("5 * var * {\n              id: int64,\n              name: string,\n              description: {\n                languages: var * string[2],\n                texts: json,\n              },\n              status: string,\n              funded_amount: float64,\n              basket_amount: json,\n              paid_amount: json,\n              image: {\n                id: int64,\n                template_id: int64,\n              },\n              video: json,\n              activity: string,\n              sector: string,\n              use: string,\n              delinquent: bool,\n              location: {\n                country_code: string[2],\n                country: string,\n                town: json,\n                geo: {\n                  level: string,\n                  pairs: string,\n                  type: string,\n                },\n              },\n              partner_id: int64,\n              posted_date: json,\n              planned_expiration_date: json,\n              loan_amount: float64,\n              currency_exchange_loss_amount: json,\n              borrowers: var * {\n                first_name: string,\n                last_name: string,\n                gender: string[1],\n                pictured: bool,\n              },\n              terms: {\n                disbursal_date: json,\n                disbursal_currency: string[3,'A'],\n                disbursal_amount: float64,\n                loan_amount: float64,\n                local_payments: var * {\n                  due_date: json,\n                  amount: float64,\n                },\n                scheduled_payments: var * {\n                  due_date: json,\n                  amount: float64,\n                },\n                loss_liability: {\n                  nonpayment: string,\n                  currency_exchange: string,\n                  currency_exchange_coverage_rate: json,\n                },\n              },\n              payments: var * {\n                amount: float64,\n                local_amount: float64,\n                processed_date: json,\n                settlement_date: json,\n                rounded_local_amount: float64,\n                currency_exchange_loss_amount: float64,\n                payment_id: int64,\n                comment: json,\n              },\n              funded_date: json,\n              paid_date: json,\n              journal_totals: {\n                entries: int64,\n                bulkEntries: int64,\n              },\n            }\n        ", self.sym)
        self.assertEqual(type(ds[-1]), ct.Record)
        self.assertEqual(len(ds[-1].names), 25)

    def test_strings_in_ds(self):
        ds = parse("5 * var * {\n              id: int64,\n             'my field': string,\n              name: string }\n             ", self.sym)
        self.assertEqual(len(ds[-1].names), 3)
        ds = parse('2 * var * {\n             "AASD @#$@#$ \' sdf": string,\n              id: float32,\n              id2: int64,\n              name: string }\n             ', self.sym)
        self.assertEqual(len(ds[-1].names), 4)

    def test_struct_errors(self):
        self.assertRaises(datashape.DataShapeSyntaxError, parse, '{id: int64, name: string amount: invalidtype}', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError, parse, '{id: int64, name: string, amount: invalidtype}', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError, parse, '{id: int64, name: string, amount: %}', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError, parse, '{\n' + '   id: int64;\n' + '   name: string;\n' + '   amount+ float32;\n' + '}\n', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError, parse, '{\n' + '   id: int64;\n' + "   'my field 1': string;\n" + '   amount+ float32;\n' + '}\n', self.sym)
        self.assertRaises(datashape.DataShapeSyntaxError, parse, '{\n' + '   id: int64,\n' + "   u'my field 1': string,\n" + '   amount: float32\n' + '}\n', self.sym)