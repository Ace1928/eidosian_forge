This injects a custom create_tags method onto the ec2 service resource

    This is needed because the resource model is not able to express
    creating multiple tag resources based on the fact you can apply a set
    of tags to multiple ec2 resources.
    