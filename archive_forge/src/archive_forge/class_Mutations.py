import datetime
import graphene
from graphql.utilities import print_schema
class Mutations(graphene.ObjectType):
    set_datetime = SetDatetime.Field()