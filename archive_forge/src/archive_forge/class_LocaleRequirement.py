class LocaleRequirement(Requirement):
    """
    A Qualification requirement based on the Worker's location. The Worker's location is specified by the Worker to Mechanical Turk when the Worker creates his account.
    """

    def __init__(self, comparator, locale, required_to_preview=False):
        super(LocaleRequirement, self).__init__(qualification_type_id='00000000000000000071', comparator=comparator, integer_value=None, required_to_preview=required_to_preview)
        self.locale = locale

    def get_as_params(self):
        params = {'QualificationTypeId': self.qualification_type_id, 'Comparator': self.comparator}
        if self.comparator in ('In', 'NotIn'):
            for i, locale in enumerate(self.locale, 1):
                if isinstance(locale, tuple):
                    params['LocaleValue.%d.Country' % i] = locale[0]
                    params['LocaleValue.%d.Subdivision' % i] = locale[1]
                else:
                    params['LocaleValue.%d.Country' % i] = locale
        elif isinstance(self.locale, tuple):
            params['LocaleValue.Country'] = self.locale[0]
            params['LocaleValue.Subdivision'] = self.locale[1]
        else:
            params['LocaleValue.Country'] = self.locale
        if self.required_to_preview:
            params['RequiredToPreview'] = 'true'
        return params