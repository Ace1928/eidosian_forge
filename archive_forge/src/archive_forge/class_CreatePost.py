import graphene
class CreatePost(graphene.Mutation):

    class Arguments:
        text = graphene.String(required=True)
    result = graphene.Field(CreatePostResult)

    def mutate(self, info, text):
        result = Success(yeah='yeah')
        return CreatePost(result=result)