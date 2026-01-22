def addToFilter(self, filterElement):
    """
        Method to add, remove, or skip a filter element
        """
    filtercopy = list(self.TrustRegionFilter)
    for fe in filtercopy:
        acceptableMeasure = fe.compare(filterElement)
        if acceptableMeasure == 1:
            self.TrustRegionFilter.remove(fe)
        elif acceptableMeasure == -1:
            return
    self.TrustRegionFilter.append(filterElement)