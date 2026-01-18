from oslo_reports.views.text import header as header_views
def with_type(gen):

    def newgen():
        res = gen()
        try:
            res.set_current_view_type(self.output_type)
        except AttributeError:
            pass
        return res
    return newgen