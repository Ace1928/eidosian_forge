from ..utils import SchemaBase
class DatumType:
    """An object to assist in building Vega-Lite Expressions"""

    def __repr__(self):
        return 'datum'

    def __getattr__(self, attr):
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError(attr)
        return GetAttrExpression('datum', attr)

    def __getitem__(self, attr):
        return GetItemExpression('datum', attr)

    def __call__(self, datum, **kwargs):
        """Specify a datum for use in an encoding"""
        return dict(datum=datum, **kwargs)