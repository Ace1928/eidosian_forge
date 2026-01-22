from docutils import languages, ApplicationError, TransformSpec
def apply_transforms(self):
    """Apply all of the stored transforms, in priority order."""
    self.document.reporter.attach_observer(self.document.note_transform_message)
    while self.transforms:
        if not self.sorted:
            self.transforms.sort()
            self.transforms.reverse()
            self.sorted = 1
        priority, transform_class, pending, kwargs = self.transforms.pop()
        transform = transform_class(self.document, startnode=pending)
        transform.apply(**kwargs)
        self.applied.append((priority, transform_class, pending, kwargs))