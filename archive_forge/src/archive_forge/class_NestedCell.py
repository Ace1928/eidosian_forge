class NestedCell(Cell):

    def __str__(self):
        return '{{%s}}' % self.to_string()