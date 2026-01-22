from tempest.lib.cli import output_parser
class ListEntryModel(Model):

    def __init__(self, fields, values):
        for k, v in zip(fields, values):
            setattr(self, k, v)