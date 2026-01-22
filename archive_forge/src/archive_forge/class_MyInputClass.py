import graphene
class MyInputClass(graphene.InputObjectType):

    @classmethod
    def __init_subclass_with_meta__(cls, container=None, _meta=None, fields=None, **options):
        if _meta is None:
            _meta = graphene.types.inputobjecttype.InputObjectTypeOptions(cls)
        _meta.fields = fields
        super(MyInputClass, cls).__init_subclass_with_meta__(container=container, _meta=_meta, **options)